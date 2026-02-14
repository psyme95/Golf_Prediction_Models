library(biomod2)
library(tidyverse)
library(readxl)

setwd("C:/Projects/Golf/Weekly_Modelling")

# ============================================
# CONFIG
# ============================================
TRAIN_SEASONS <- c(2024, 2025)
CURRENT_SEASON <- "S26"
NUMBER_OF_RUNS <- 30
CV_PERCENTAGE <- 0.8
CV_REPETITIONS <- 5
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')

model_vars <- c("Rd2Pos", "Rd2Lead", "AvPosn", "Top5", 
                "GLM_Odds_Probability_Median", "Model_Score_Median")

# ============================================
# TRAINING FUNCTION
# ============================================
train_ensemble_with_calibration <- function(train_data, model_vars, tour_name, run_number) {
  
  resp <- train_data$Market_Win
  expl <- train_data %>% select(all_of(model_vars)) %>% as.data.frame()
  
  myBiomodData <- BIOMOD_FormatingData(
    resp.var = resp,
    expl.var = expl,
    resp.name = paste0(tour_name, "_R2")
  )
  
  myBiomodData@coord <- data.frame(
    x = seq_len(length(myBiomodData@data.species)),
    y = seq_len(length(myBiomodData@data.species))
  )
  
  user.rf <- list('_allData_allRun' = list(
    ntree = 1500, nodesize = 350, mtry = floor(sqrt(length(model_vars)))
  ))
  
  user.XGBOOST <- list('_allData_allRun' = list(
    nrounds = 50, subsample = 0.5, colsample_bytree = 0.5,
    min_child_weight = 5, print_every_n = 10L
  ))
  
  user.GAM <- list('_allData_allRun' = list(
    algo = 'GAM_mgcv', type = 's_smoother', k = 6, interaction.level = 2,
    myFormula = NULL, family = binomial(link = 'logit'), method = 'REML',
    optimizer = c('outer', 'newton'), select = TRUE, knots = NULL,
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
  
  myBiomodModelEval <- get_evaluations(myBiomodModelOut)
  
  cv.eval <- myBiomodModelEval %>%
    filter(run != "allRun", metric.eval == "TSS") %>%
    group_by(full.name) %>%
    summarise(
      mean_sensitivity = mean(sensitivity),
      mean_specificity = mean(specificity),
      .groups = "drop"
    ) %>%
    mutate(TSS = (mean_sensitivity / 100 + mean_specificity / 100) - 1)
  
  allRun.eval <- myBiomodModelEval %>%
    filter(run == "allRun", metric.eval == "TSS")
  
  df.metrics <- data.frame(matrix(0, ncol = nrow(allRun.eval), nrow = 1))
  names(df.metrics) <- allRun.eval$full.name
  row.names(df.metrics) <- "TSS"
  
  for (model_name in allRun.eval$full.name) {
    matching_cv <- cv.eval %>%
      filter(grepl(gsub(".*_allRun_", "", model_name), full.name))
    if (nrow(matching_cv) > 0) {
      df.metrics[1, model_name] <- mean(matching_cv$TSS)
    }
  }
  
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
  
  model_predictions <- get_predictions(myBiomodModelOut)
  cv_predictions <- model_predictions %>% filter(run != "allRun")
  
  model_weights <- as.numeric(df.metrics[1, ])
  model_weights <- model_weights / sum(model_weights)
  names(model_weights) <- names(df.metrics)
  
  n_obs <- nrow(train_data)
  oof_scores <- rep(NA_real_, n_obs)
  oof_counts <- rep(0, n_obs)
  
  for (cv_run in unique(cv_predictions$run)) {
    run_preds <- cv_predictions %>% filter(run == cv_run)
    run_models <- unique(run_preds$full.name)
    
    fold_ensemble <- numeric(n_obs)
    fold_has_pred <- logical(n_obs)
    
    for (model_name in run_models) {
      model_preds <- run_preds %>% filter(full.name == model_name)
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
  
  if (mean(!is.na(oof_scores) & oof_counts > 0) < 1) {
    insample_scores <- myBiomodEM@models.prediction@val$pred
    missing_idx <- is.na(oof_scores) | oof_counts == 0
    oof_scores[missing_idx] <- insample_scores[missing_idx]
  }
  
  cal_df <- data.frame(
    y = train_data$Market_Win,
    oof_scores = oof_scores,
    odds = train_data$Betfair_rd2
  )
  calibration_model <- glm(y ~ oof_scores + odds, data = cal_df, family = binomial())
  
  list(
    ensemble_model = myBiomodEM,
    individual_models = myBiomodModelOut,
    calibration_model = calibration_model,
    metrics = df.metrics,
    model_weights = model_weights
  )
}

# ============================================
# TRAIN AND SAVE
# ============================================
train_and_save_models <- function(tour_data, tour_name, train_seasons) {
  
  cat("\n", strrep("=", 60), "\n")
  cat("Training", tour_name, "on seasons:", paste(train_seasons, collapse = ", "), "\n")
  cat(strrep("=", 60), "\n")
  
  train_data <- tour_data %>%
    filter(Test_Year %in% train_seasons) %>%
    rename(Top5 = `_Top5`)
  
  cat("Training rows:", nrow(train_data), "| Wins:", sum(train_data$Market_Win), "\n")
  
  run_results <- list()
  
  for (run in 1:NUMBER_OF_RUNS) {
    cat("\n--- Run", run, "of", NUMBER_OF_RUNS, "---\n")
    run_results[[run]] <- train_ensemble_with_calibration(
      train_data, model_vars, tour_name, run
    )
  }
  
  model_package <- list(
    run_results = run_results,
    model_vars = model_vars,
    train_seasons = train_seasons,
    tour_name = tour_name,
    trained_at = Sys.time(),
    n_runs = NUMBER_OF_RUNS
  )
  
  output_dir <- "./Output/Models/"
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  output_path <- file.path(output_dir, paste0(tour_name, "_R2_", CURRENT_SEASON, "_trained.rds"))
  saveRDS(model_package, output_path)
  
  cat("\nModel saved to:", output_path, "\n")
  
  model_package
}

# ============================================
# MAIN
# ============================================

euro_preds <- read_excel("./Input/Full_Euro_Historical_Predictions.xlsx")
pga_preds <- read_excel("./Input/Full_PGA_Historical_Predictions.xlsx")
euro <- read_excel("./Input/Euro.xlsx")
pga <- read_excel("./Input/PGA.xlsx")

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

pga_model <- train_and_save_models(pga_preds, "PGA", TRAIN_SEASONS)
euro_model <- train_and_save_models(euro_preds, "Euro", TRAIN_SEASONS)