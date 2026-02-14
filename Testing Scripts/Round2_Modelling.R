library(biomod2)
library(tidyverse)
library(readxl)
library(pROC)
library(openxlsx)
library(here)

setwd("C:/Projects/Golf")

# ============================================
# CONFIG
# ============================================
MODEL_SUFFIX <- "R2"
TRAIN_WINDOW <- 2
NUMBER_OF_RUNS <- 5
CV_PERCENTAGE <- 0.7
CV_REPETITIONS <- 3
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')

model_vars <- c("Rd2Pos", "Rd2Lead", "AvPosn", "Top5", 
                "GLM_Odds_Probability_Median", "Model_Score_Median")

# ============================================
# FUNCTIONS
# ============================================
train_ensemble_with_calibration <- function(train_data, model_vars, tour_name, run_number) {
  
  resp <- train_data$Market_Win
  expl <- train_data %>% select(all_of(model_vars)) %>% as.data.frame()
  
  myBiomodData <- BIOMOD_FormatingData(
    resp.var = resp,
    expl.var = expl,
    resp.name = paste0(tour_name, "_R2")
  )
  
  dummy_coords <- data.frame(
    x = seq_len(length(myBiomodData@data.species)),
    y = seq_len(length(myBiomodData@data.species))
  )
  myBiomodData@coord <- dummy_coords
  
  # Biomod options
  user.rf <- list('_allData_allRun' = list(
    ntree = 1500,
    nodesize = 350,
    mtry = floor(sqrt(length(model_vars)))
  ))
  
  user.XGBOOST <- list('_allData_allRun' = list(
    nrounds = 50,
    subsample = 0.5,
    colsample_bytree = 0.5,
    min_child_weight = 5,
    print_every_n = 10L
  ))
  
  user.GAM <- list('_allData_allRun' = list(
    algo = 'GAM_mgcv',
    type = 's_smoother',
    k = 6,
    interaction.level = 2,
    myFormula = NULL,
    family = binomial(link = 'logit'),
    method = 'REML',
    optimizer = c('outer', 'newton'),
    select = TRUE,
    knots = NULL,
    paraPen = NULL,
    control = list(
      nthreads = 1, irls.reg = 0, epsilon = 1e-07, maxit = 200,
      trace = FALSE, mgcv.tol = 1e-07, mgcv.half = 15,
      rank.tol = 1.49011611938477e-08,
      nlm = list(ndigit = 7, gradtol = 1e-06, stepmax = 2, 
                 steptol = 1e-04, iterlim = 200, check.analyticals = 0),
      optim = list(factr = 1e+07),
      newton = list(conv.tol = 1e-06, maxNstep = 5, maxSstep = 2,
                    maxHalf = 30, use.svd = 0),
      idLinksBases = TRUE, scalePenalty = TRUE, keepData = FALSE
    )
  ))
  
  user.val <- list(
    RF.binary.randomForest.randomForest = user.rf,
    XGBOOST.binary.xgboost.xgboost = user.XGBOOST,
    GAM.binary.mgcv.gam = user.GAM
  )
  
  myOpt <- bm_ModelingOptions(
    data.type = 'binary',
    models = MODELS_TO_PROCESS,
    strategy = "user.defined",
    user.val = user.val,
    bm.format = myBiomodData
  )
  
  # Train models
  myBiomodModelOut <- BIOMOD_Modeling(
    bm.format = myBiomodData,
    modeling.id = paste0("Run", run_number),
    models = MODELS_TO_PROCESS,
    OPT.user = myOpt,
    OPT.user.val = user.val,
    CV.nb.rep = CV_REPETITIONS,
    CV.perc = CV_PERCENTAGE,
    var.import = 1,
    metric.eval = c('TSS', 'ROC'),
    nb.cpu = 1,
    do.progress = TRUE
  )
  
  # Get evaluations - use CV runs for honest metrics
  myBiomodModelEval <- get_evaluations(myBiomodModelOut)
  
  # Use CV run evaluations for ensemble weights (not allRun)
  cv.eval <- myBiomodModelEval %>%
    filter(run != "allRun", metric.eval == "TSS") %>%
    group_by(full.name) %>%
    summarise(
      mean_sensitivity = mean(sensitivity),
      mean_specificity = mean(specificity),
      .groups = "drop"
    ) %>%
    mutate(TSS = (mean_sensitivity / 100 + mean_specificity / 100) - 1)
  
  # Build metrics dataframe for ensemble using CV-based TSS
  allRun.eval <- myBiomodModelEval %>%
    filter(run == "allRun", metric.eval == "TSS")
  
  df.metrics <- data.frame(matrix(0, ncol = nrow(allRun.eval), nrow = 1))
  names(df.metrics) <- allRun.eval$full.name
  row.names(df.metrics) <- "TSS"
  
  # Match CV metrics to allRun model names
  for (model_name in allRun.eval$full.name) {
    matching_cv <- cv.eval %>%
      filter(grepl(gsub(".*_allRun_", "", model_name), full.name))
    
    if (nrow(matching_cv) > 0) {
      df.metrics[1, model_name] <- mean(matching_cv$TSS)
    }
  }
  
  cat("    CV-based model TSS:", round(as.numeric(df.metrics[1, ]), 3), "\n")
  
  # Create ensemble
  myBiomodEM <- BIOMOD_EnsembleModeling(
    bm.mod = myBiomodModelOut,
    models.chosen = names(df.metrics),
    em.algo = c('EMwmean'),
    metric.select = c("user.defined"),
    metric.select.thresh = c(0.1),
    metric.select.table = df.metrics,
    metric.eval = c('TSS', 'ROC'),
    var.import = 1,
    EMwmean.decay = 'proportional'
  )
  
  # ============================================
  # OUT-OF-FOLD PREDICTIONS FOR CALIBRATION
  # ============================================
  
  model_predictions <- get_predictions(myBiomodModelOut)
  
  cv_predictions <- model_predictions %>%
    filter(run != "allRun")
  
  # Get model weights from ensemble (proportional TSS weighting)
  model_weights <- as.numeric(df.metrics[1, ])
  model_weights <- model_weights / sum(model_weights)
  names(model_weights) <- names(df.metrics)
  
  # Build OOF ensemble predictions
  n_obs <- nrow(train_data)
  oof_scores <- rep(NA_real_, n_obs)
  oof_counts <- rep(0, n_obs)
  
  cv_runs <- unique(cv_predictions$run)
  
  for (cv_run in cv_runs) {
    
    run_preds <- cv_predictions %>%
      filter(run == cv_run)
    
    run_models <- unique(run_preds$full.name)
    
    fold_ensemble <- numeric(n_obs)
    fold_has_pred <- logical(n_obs)
    
    for (model_name in run_models) {
      
      model_preds <- run_preds %>%
        filter(full.name == model_name)
      
      allRun_name <- gsub(paste0("_", cv_run), "_allRun", model_name)
      
      if (allRun_name %in% names(model_weights)) {
        weight <- model_weights[allRun_name]
        
        pred_indices <- which(!is.na(model_preds$pred))
        fold_ensemble[pred_indices] <- fold_ensemble[pred_indices] + 
          weight * model_preds$pred[pred_indices]
        fold_has_pred[pred_indices] <- TRUE
      }
    }
    
    total_weight <- sum(model_weights[gsub(paste0("_", cv_run), "_allRun", run_models)])
    fold_ensemble <- fold_ensemble / total_weight
    
    oof_scores[fold_has_pred] <- ifelse(
      is.na(oof_scores[fold_has_pred]),
      fold_ensemble[fold_has_pred],
      oof_scores[fold_has_pred] + fold_ensemble[fold_has_pred]
    )
    oof_counts[fold_has_pred] <- oof_counts[fold_has_pred] + 1
  }
  
  oof_scores <- oof_scores / pmax(oof_counts, 1)
  
  coverage <- mean(!is.na(oof_scores) & oof_counts > 0)
  cat("    OOF prediction coverage:", round(coverage * 100, 1), "%\n")
  
  if (coverage < 1) {
    warning("Some observations lack OOF predictions - using in-sample for these")
    insample_scores <- myBiomodEM@models.prediction@val$pred
    missing_idx <- is.na(oof_scores) | oof_counts == 0
    oof_scores[missing_idx] <- insample_scores[missing_idx]
  }
  
  # ============================================
  # FIT CALIBRATION ON OOF PREDICTIONS
  # ============================================
  
  training_labels <- train_data$Market_Win
  training_odds <- train_data$Betfair_rd2
  
  cat("    Fitting GLM calibration on OOF predictions...\n")
  
  calibration_model <- glm(
    training_labels ~ oof_scores + training_odds,
    family = binomial()
  )
  
  # Diagnostic: compare OOF vs in-sample calibration
  insample_scores <- myBiomodEM@models.prediction@val$pred
  cat("    Correlation OOF vs in-sample scores:", 
      round(cor(oof_scores, insample_scores), 3), "\n")
  
  return(list(
    ensemble_model = myBiomodEM,
    individual_models = myBiomodModelOut,
    calibration_model = calibration_model,
    metrics = df.metrics,
    model_eval = myBiomodModelEval,
    training_size = nrow(train_data),
    oof_scores = oof_scores
  ))
}

# Apply calibration to get probabilities
apply_calibration <- function(scores, odds, calibration_model) {
  new_data <- data.frame(oof_scores = scores, training_odds = odds)
  probs <- predict(calibration_model, newdata = new_data, type = "response")
  return(probs)
}

# Run single test year with multiple runs
run_test_year <- function(tour_data, tour_name, test_year) {
  
  cat("\n", strrep("=", 60), "\n")
  cat(tour_name, "- Test Year:", test_year, "\n")
  cat(strrep("=", 60), "\n")
  
  # Split data
  train_data <- tour_data %>%
    filter(Test_Year %in% (test_year - TRAIN_WINDOW):(test_year - 1)) %>%
    rename(Top5 = `_Top5`)
  
  test_data <- tour_data %>%
    filter(Test_Year == test_year) %>%
    rename(Top5 = `_Top5`)
  
  cat("Training:", nrow(train_data), "rows |", sum(train_data$Market_Win), "wins\n")
  cat("Test:", nrow(test_data), "rows |", sum(test_data$Market_Win), "wins\n")
  
  # Run multiple times
  run_results <- list()
  
  for (run in 1:NUMBER_OF_RUNS) {
    cat("\n--- Run", run, "of", NUMBER_OF_RUNS, "---\n")
    
    result <- train_ensemble_with_calibration(
      train_data = train_data,
      model_vars = model_vars,
      tour_name = tour_name,
      run_number = run
    )
    
    run_results[[run]] <- result
  }
  
  # Get predictions from each run and average
  cat("\nGenerating test predictions...\n")
  
  test_expl <- test_data %>% select(all_of(model_vars)) %>% as.data.frame()
  test_coords <- data.frame(x = seq_len(nrow(test_data)), y = seq_len(nrow(test_data)))
  
  all_scores <- matrix(NA, nrow = nrow(test_data), ncol = NUMBER_OF_RUNS)
  all_calibrated_probs <- matrix(NA, nrow = nrow(test_data), ncol = NUMBER_OF_RUNS)
  
  for (run in 1:NUMBER_OF_RUNS) {
    
    test_proj <- BIOMOD_EnsembleForecasting(
      bm.em = run_results[[run]]$ensemble_model,
      bm.proj = NULL,
      proj.name = paste0("Test_Run", run),
      new.env = test_expl,
      new.env.xy = test_coords,
      models.chosen = "all"
    )
    
    scores <- test_proj@proj.out@val$pred
    all_scores[, run] <- scores
    
    calibrated <- apply_calibration(
      scores = scores,
      odds = test_data$Betfair_rd2,
      calibration_model = run_results[[run]]$calibration_model
    )
    
    all_calibrated_probs[, run] <- calibrated
  }
  
  # Average across runs
  avg_scores <- rowMeans(all_scores)
  avg_calibrated_probs <- rowMeans(all_calibrated_probs)
  
  # ============================================
  # EVALUATION
  # ============================================
  
  roc_obj <- roc(test_data$Market_Win, avg_calibrated_probs, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj))
  
  coords_best <- coords(roc_obj, "best", ret = c("threshold", "sensitivity", "specificity"))
  tss_test <- coords_best$sensitivity + coords_best$specificity - 1
  
  brier <- mean((avg_calibrated_probs - test_data$Market_Win)^2)
  
  cat("\n--- TEST SET RESULTS ---\n")
  cat("ROC-AUC:", round(auc_val, 3), "\n")
  cat("TSS (optimal threshold):", round(tss_test, 3), "\n")
  cat("Brier Score:", round(brier, 4), "\n")
  
  # Results dataframe with normalised probability
  results <- test_data %>%
    mutate(
      Rd2_Model_Score_Avg = avg_scores,
      Rd2_Calibrated_Prob = avg_calibrated_probs,
      Rd2_Model_Odds = 1 / avg_calibrated_probs,
    ) %>%
    group_by(EventID) %>%
    mutate(
      Rd2_Normalised_Prob = Rd2_Calibrated_Prob / sum(Rd2_Calibrated_Prob),
      Normalised_Odds = 1 / Rd2_Normalised_Prob
    ) %>%
    ungroup()
  
  return(list(
    results = results,
    run_results = run_results,
    auc = auc_val,
    tss = tss_test,
    brier = brier,
    tour = tour_name,
    test_year = test_year
  ))
}

# ============================================
# MAIN EXECUTION
# ============================================
run_all_models <- function(tour_data, tour_name) {
  
  available_years <- sort(unique(tour_data$Test_Year))
  min_year <- min(available_years)
  
  test_years <- available_years[available_years >= min_year + TRAIN_WINDOW]
  
  cat("\nTour:", tour_name, "\n")
  cat("Available years:", paste(available_years, collapse = ", "), "\n")
  cat("Test years:", paste(test_years, collapse = ", "), "\n")
  
  all_results <- list()
  
  for (yr in test_years) {
    result <- run_test_year(tour_data, tour_name, yr)
    all_results[[as.character(yr)]] <- result
  }
  
  # Summary
  cat("\n", strrep("=", 60), "\n")
  cat("SUMMARY -", tour_name, "\n")
  cat(strrep("=", 60), "\n")
  
  summary_df <- map_dfr(all_results, ~tibble(
    Year = .x$test_year,
    AUC = .x$auc,
    TSS = .x$tss,
    Brier = .x$brier
  ))
  
  print(summary_df)
  cat("\nMean AUC:", round(mean(summary_df$AUC), 3), "\n")
  cat("Mean TSS:", round(mean(summary_df$TSS), 3), "\n")
  cat("Mean Brier:", round(mean(summary_df$Brier), 4), "\n")
  
  return(all_results)
}

# ============================================
# LOAD DATA AND EXECUTE
# ============================================

# Load data
euro_preds <- read_excel(here("Data", "Full_Euro_Historical_Predictions.xlsx"))
pga_preds <- read_excel(here("Data", "Full_PGA_Historical_Predictions.xlsx"))
euro <- read_excel(here("Data", "Euro.xlsx"))
pga <- read_excel(here("Data", "PGA.xlsx"))

# Tidy and join
euro_clean <- euro %>%
  select(eventID, playerID, Rd2Pos, Rd2Lead, AvPosn, `_Top5`, Betfair_rd2) %>%
  drop_na()

pga_clean <- pga %>%
  select(eventID, playerID, Rd2Pos, Rd2Lead, AvPosn, `_Top5`, Betfair_rd2) %>%
  drop_na()

pga_preds <- pga_preds %>%
  left_join(pga_clean, by = c("EventID" = "eventID", "PlayerID" = "playerID")) %>%
  drop_na()

euro_preds <- euro_preds %>%
  left_join(euro_clean, by = c("EventID" = "eventID", "PlayerID" = "playerID")) %>%
  drop_na()

# Run models
pga_results <- run_all_models(pga_preds, "PGA")
euro_results <- run_all_models(euro_preds, "Euro")

# ============================================
# EXPORT RESULTS
# ============================================
export_results <- function(results_list, tour_name) {
  
  output_dir <- "./Output/R2_Models/"
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  # Combine all year results
  all_predictions <- map_dfr(results_list, ~.x$results)
  
  # Save predictions
  write.xlsx(
    all_predictions,
    paste0(output_dir, tour_name, "_R2_Predictions_", MODEL_SUFFIX, ".xlsx")
  )
  
  # Save models
  saveRDS(results_list, paste0(output_dir, tour_name, "_R2_Models_", MODEL_SUFFIX, ".rds"))
  
  cat("\nResults exported for", tour_name, "\n")
}

export_results(pga_results, "PGA")
export_results(euro_results, "Euro")
