# ===== CONFIGURATION SECTION =====
# TESTING NEW PARAMETERS FOR XGBOOST AND RF

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
# MODELS_TO_PROCESS <- c('XGBOOST', 'GAM', 'RF')
# MODELS_TO_PROCESS <- c('RF', 'GAM')

# Cross-validation settings
USE_CROSS_VALIDATION <- TRUE  # Enable cross-validation
CV_STRATEGY <- "kfold"  # Use k-fold cross-validation (better for rare events)
CV_K_FOLDS <- 5  # Number of folds for k-fold CV
CV_REPETITIONS <- 1  # Number of times to repeat k-fold (1 is usually sufficient)
CV_PERCENTAGE <- 0.8

MODEL_NAME <- paste0("WinMarket_1YrTrain_", format(Sys.time(), "%d%m_%H%M"))
NUM_ITERATIONS <- 10  # Number of times to repeat the entire process
start_time <- Sys.time()

# Define betting markets to test
BETTING_MARKETS <- list(
  "Winner" = list(
    target_col = "win",
    odds_col = "Win_odds",
    position_threshold = 1,
    model_odds_cols = c("Win_odds", "Top5_odds")  # Winner + next level
  )
)

# ===== LOAD REQUIRED PACKAGES =====
library(biomod2)
library(dplyr)
library(readxl)
library(lubridate)
library(ggplot2)
library(here)

# Define 'not in' operator
`%not in%` <- Negate(`%in%`)

# ===== HELPER FUNCTIONS =====
# Platt Scaling Functions
apply_platt_scaling <- function(scores, labels) {
  # Fit sigmoid parameters A and B using maximum likelihood
  sigmoid <- function(x, A, B) {
    1 / (1 + exp(A * x + B))
  }
  
  # Objective function to minimize (negative log-likelihood)
  neg_log_likelihood <- function(params) {
    A <- params[1]
    B <- params[2]
    probs <- sigmoid(scores, A, B)
    # Add small epsilon to avoid log(0)
    probs <- pmax(pmin(probs, 1 - 1e-15), 1e-15)
    -sum(labels * log(probs) + (1 - labels) * log(1 - probs))
  }
  
  # Optimize parameters
  result <- optim(c(0, 0), neg_log_likelihood, method = "BFGS")
  
  return(list(
    A = result$par[1], 
    B = result$par[2]
  ))
}

# Apply Platt scaling to new scores using fitted parameters
predict_platt_scaling <- function(scores, platt_params) {
  1 / (1 + exp(platt_params$A * scores + platt_params$B))
}

# Model training function (adapted for any target variable)
train_ensemble_model <- function(train_data, model_vars, target_variable) {
  resp <- train_data[, target_variable]
  expl <- train_data[, model_vars]
  
  myBiomodData <- BIOMOD_FormatingData(
    resp.var = resp,
    expl.var = expl,
    resp.name = "PGA"
  )
  
  # Create dummy coordinates
  dummy_coords <- data.frame(
    x = 1:length(myBiomodData@data.species),
    y = 1:length(myBiomodData@data.species)
  )
  myBiomodData@coord <- dummy_coords
  
  # Biomod options
  user.rf <- list('_allData_allRun' = list(ntree=2000,
                                           nodesize=1,
                                           maxnodes = 20,
                                           mtry = 2,
                                           classwt = c("0"=1, "1"=1000)
  ))
  user.XGBOOST <- list('_allData_allRun' = list(params = list(eta = 0.05,
                                                              max_depth = 3,
                                                              lambda = 10,
                                                              alpha = 2),
                                                nrounds = 50,
                                                subsample = 0.5,
                                                colsample_bytree = 0.5,
                                                min_child_weight = 5,
                                                print_every_n = 10L
  ))
  user.GAM <- list('_allData_allRun' = list( algo = 'GAM_mgcv',
                                             type = 's_smoother',
                                             k = 6,
                                             interaction.level = 2,
                                             myFormula = NULL,
                                             family = binomial(link = 'logit'),
                                             method = 'REML',
                                             optimizer = c('outer','newton'),
                                             select = TRUE,
                                             knots = NULL,
                                             paraPen = NULL,
                                             control = list(nthreads = 1, irls.reg = 0, epsilon = 1e-07, maxit = 200, 
                                                            trace = FALSE, mgcv.tol = 1e-07, mgcv.half = 15,
                                                            rank.tol = 1.49011611938477e-08, nlm = list(ndigit=7,
                                                                                                        gradtol=1e-06, stepmax=2, steptol=1e-04, iterlim=200,
                                                                                                        check.analyticals=0), optim = list(factr=1e+07), 
                                                            newton = list(conv.tol=1e-06, maxNstep=5, maxSstep=2,
                                                                          maxHalf=30, use.svd=0),
                                                            idLinksBases = TRUE, scalePenalty = TRUE, keepData = FALSE)))
  user.GLM <- list('_allData_allRun' = list(type = "quadratic",
                                            interaction.level=1))
  user.SRE <- list('_allData_allRun' = list(quant=0.025))
  user.MARS <- list('_allData_allRun' = list(type='quadratic',
                                             interaction.level = 1))
  user.CTA <- list('_allData_allRun' = list(control = rpart.control(cp=0.005)))
  
  user.val <- list(RF.binary.randomForest.randomForest = user.rf,
                   XGBOOST.binary.xgboost.xgboost = user.XGBOOST,
                   GAM.binary.mgcv.gam = user.GAM,
                   GLM.binary.stats.glm = user.GLM,
                   SRE.binary.biomod2.bm_SRE = user.SRE,
                   MARS.binary.earth.earth = user.MARS,
                   CTA.binary.rpart.rpart = user.CTA)
  
  myOpt <- bm_ModelingOptions(data.type = 'binary',
                              models=MODELS_TO_PROCESS,
                              strategy = "user.defined",
                              user.val = user.val,
                              bm.format = myBiomodData)
  
  # Train individual models
  myBiomodModelOut <- BIOMOD_Modeling(
    bm.format = myBiomodData,
    models = MODELS_TO_PROCESS,
    OPT.user = myOpt,
    OPT.user.val = user.val,
    CV.nb.rep = CV_REPETITIONS,
    CV.perc = CV_PERCENTAGE,
    weights = NULL,
    var.import = 1,
    prevalence = 0.9,
    metric.eval = c('TSS'),
    nb.cpu = 1,
    do.progress = TRUE
  )
  
  # Evaluate models
  myBiomodModelEval <- get_evaluations(myBiomodModelOut)
  allRun.model.names <- myBiomodModelEval[
    which(myBiomodModelEval$run == "allRun" & myBiomodModelEval$metric.eval == "TSS"),
    "full.name"
  ]
  allRun.model.sens <- myBiomodModelEval[
    which(myBiomodModelEval$run == "allRun" & myBiomodModelEval$metric.eval == "TSS"),
    "sensitivity"
  ]
  allRun.model.spec <- myBiomodModelEval[
    which(myBiomodModelEval$run == "allRun" & myBiomodModelEval$metric.eval == "TSS"),
    "specificity"
  ]
  allRun.model.TSS <- (allRun.model.sens / 100 + allRun.model.spec / 100) - 1
  
  # Create metrics dataframe
  df.metrics <- data.frame(matrix(0, ncol = length(allRun.model.TSS), nrow = 1))
  names(df.metrics) <- allRun.model.names
  row.names(df.metrics) <- "TSS"
  df.metrics[1, 1:length(allRun.model.TSS)] <- allRun.model.TSS
  
  # Create ensemble model
  myBiomodEM <- BIOMOD_EnsembleModeling(
    bm.mod = myBiomodModelOut,
    models.chosen = names(df.metrics),
    em.algo = c('EMwmean'),
    metric.select = c("user.defined"),
    metric.select.thresh = c(0.1),
    metric.select.table = df.metrics,
    metric.eval = c('TSS'),
    var.import = 1,
    EMwmean.decay = 'proportional'
  )
  
  # Store model evaluation metrics
  model_metrics <- data.frame(
    model_name = allRun.model.names,
    TSS = allRun.model.TSS,
    sensitivity = allRun.model.sens,
    specificity = allRun.model.spec,
    cv_strategy = CV_STRATEGY,
    cv_k_folds = CV_K_FOLDS,
    cv_repetitions = CV_REPETITIONS,
    model_type = "allRun"  # Full data model
  )
  
  return(list(
    ensemble = myBiomodEM,
    metrics = model_metrics
  ))
}

# Function to get market-specific model variables
get_market_model_vars <- function(market_config, base_model_vars) {
  # Start with base variables (excluding all odds columns initially)
  base_vars_no_odds <- base_model_vars[!base_model_vars %in% 
                                         c("Win_odds", "Top5_odds", "Top10_odds", "Top20_odds")]
  
  # Add market-specific odds columns as predictors
  market_specific_vars <- c(base_vars_no_odds, market_config$model_odds_cols)
  
  return(market_specific_vars)
}

# Function to get year pairs for training/testing
get_year_pairs <- function(df_all) {
  # Extract years from dates
  df_all <- df_all %>%
    mutate(year = year(Date))
  
  # Get unique years, sorted
  unique_years <- sort(unique(df_all$year))
  
  # Create training/test year pairs
  # We need at least 1 year for training, so start from index 2
  year_pairs <- list()
  for (i in 2:length(unique_years)) {
    year_pairs[[i-1]] <- list(
      train_year = unique_years[i-1],  # Previous year
      test_year = unique_years[i]
    )
  }
  
  return(list(
    year_pairs = year_pairs,
    df_with_year = df_all
  ))
}

# Process single betting market with yearly validation
process_market_yearly_validation <- function(market_name, market_config, df_all, 
                                             base_model_vars, year_pairs) {
  
  cat("\n=== PROCESSING", market_name, "MARKET ===\n")
  
  # Get market-specific model variables
  market_model_vars <- get_market_model_vars(market_config, base_model_vars)
  
  # Check available variables
  missing_vars <- market_model_vars[market_model_vars %not in% names(df_all)]
  available_vars <- market_model_vars[market_model_vars %in% names(df_all)]
  
  cat("Available variables for", market_name, ":", length(available_vars), "/", 
      length(market_model_vars), "\n")
  
  # Initialize storage for all iterations
  all_iterations_results <- list()
  all_model_metrics <- list()
  
  for (iteration in 1:NUM_ITERATIONS) {
    cat("\n=== ITERATION", iteration, "OF", NUM_ITERATIONS, "FOR", market_name, "===\n")
    
    iteration_predictions_list <- list()
    iteration_metrics_list <- list()
    
    # Loop through each year pair
    for (pair_idx in seq_along(year_pairs)) {
      train_year <- year_pairs[[pair_idx]]$train_year  # Now singular
      test_year <- year_pairs[[pair_idx]]$test_year
      
      cat("\n  Training on year", train_year, 
          "-> Testing on year", test_year, "\n")
      
      # Get training data (one full year)
      train_data <- df_all %>%
        filter(year == train_year) %>%  # Use == for single year
        select(all_of(c("playerID", "eventID", "year", "posn", 
                        market_config$target_col, market_config$odds_col,
                        available_vars))) %>%
        filter(complete.cases(.))
      
      # Get test data (following year)
      test_data <- df_all %>%
        filter(year == test_year) %>%
        select(all_of(c("playerID", "eventID", "year", "posn", 
                        market_config$target_col, market_config$odds_col,
                        "rating", available_vars))) %>%
        filter(complete.cases(.))
      
      cat("  Training samples:", nrow(train_data), "| Test samples:", nrow(test_data), "\n")
      
      # Train ensemble model
      model_result <- train_ensemble_model(train_data, available_vars, market_config$target_col)
      current_model <- model_result$ensemble
      model_metrics <- model_result$metrics %>%
        mutate(
          iteration = iteration,
          train_year = train_year,  # Store single year
          test_year = test_year,
          market = market_name
        )
      iteration_metrics_list[[length(iteration_metrics_list) + 1]] <- model_metrics
      
      # Get training predictions for calibration fitting
      training_scores <- current_model@models.prediction@val$pred
      training_labels <- train_data[[market_config$target_col]]
      training_odds <- train_data[[market_config$odds_col]]
      
      # Train calibration models using training data
      training_implied_prob <- 1 / training_odds
      
      # 1. Logistic Regression Calibration (with odds)
      logit_model_data_odds <- data.frame(
        Model_Score = training_scores,
        Market_Win = training_labels,
        Market_Odds = training_odds
      )
      
      calibration_model_glm_odds <- glm(
        Market_Win ~ Model_Score + Market_Odds,
        data = logit_model_data_odds,
        family = binomial()
      )
      
      # 2. Logistic Regression Calibration (with implied probability)
      logit_model_data_prob <- data.frame(
        Model_Score = training_scores,
        Market_Win = training_labels,
        Implied_Prob = training_implied_prob
      )
      
      calibration_model_glm_prob <- glm(
        Market_Win ~ Model_Score + Implied_Prob,
        data = logit_model_data_prob,
        family = binomial()
      )
      
      # 3. Platt Scaling Calibration
      platt_params <- apply_platt_scaling(training_scores, training_labels)
      
      # Prepare test data for biomod projection
      test_expl <- test_data[, available_vars]
      test_resp.xy <- test_data[, c("playerID", "eventID")]
      
      # Run biomod projection on test year
      myBiomodProj <- BIOMOD_EnsembleForecasting(
        bm.em = current_model,
        bm.proj = NULL,
        proj.name = paste0("iter", iteration, "_", 
                           train_year, "_to_", 
                           test_year, "_", market_name),  # Updated naming
        new.env = test_expl,
        new.env.xy = test_resp.xy,
        models.chosen = "all",
        metric.binary = "TSS",
        metric.filter = "TSS",
        na.rm = TRUE
      )
      
      # Process predictions
      test_predictions <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
      colnames(test_predictions)[3] <- "Model_Score"
      
      # Merge with test data
      test_predictions <- merge(
        test_predictions,
        test_data[, c("eventID", "playerID", "year", "posn", 
                      market_config$target_col, market_config$odds_col, "rating")],
        by = c("eventID", "playerID"),
        all.x = TRUE
      )
      
      # Apply calibration methods
      test_implied_prob <- 1 / test_predictions[[market_config$odds_col]]
      
      # 1. GLM Calibration (with odds)
      calibration_data_glm_odds <- data.frame(
        Model_Score = test_predictions$Model_Score,
        Market_Odds = test_predictions[[market_config$odds_col]]
      )
      
      test_predictions$GLM_Odds_Probability <- predict(
        calibration_model_glm_odds, 
        calibration_data_glm_odds, 
        type = "response"
      )
      
      # 2. GLM Calibration (with implied probability)
      calibration_data_glm_prob <- data.frame(
        Model_Score = test_predictions$Model_Score,
        Implied_Prob = test_implied_prob
      )
      
      test_predictions$GLM_ImpliedProb_Probability <- predict(
        calibration_model_glm_prob, 
        calibration_data_glm_prob, 
        type = "response"
      )
      
      # 3. Platt Scaling Calibration
      test_predictions$Platt_Probability <- predict_platt_scaling(
        test_predictions$Model_Score, 
        platt_params
      )
      
      # Store predictions
      year_predictions <- data.frame(
        Iteration = iteration,
        Train_Year = train_year,  # Store single year
        Test_Year = test_year,
        EventID = test_predictions$eventID,
        PlayerID = test_predictions$playerID,
        Model_Score = round(test_predictions$Model_Score, 4),
        Market_Win = test_predictions[[market_config$target_col]],
        Position = test_predictions$posn,
        Market_Odds = test_predictions[[market_config$odds_col]],
        Player_Rating = test_predictions$rating,
        GLM_Odds_Probability = test_predictions$GLM_Odds_Probability,
        GLM_ImpliedProb_Probability = test_predictions$GLM_ImpliedProb_Probability,
        Platt_Probability = test_predictions$Platt_Probability
      )
      
      iteration_predictions_list[[length(iteration_predictions_list) + 1]] <- year_predictions
    }
    
    # Combine all predictions for this iteration
    if (length(iteration_predictions_list) > 0) {
      iteration_predictions <- do.call(rbind, iteration_predictions_list)
      all_iterations_results[[iteration]] <- iteration_predictions
    }
    
    # Combine all metrics for this iteration
    if (length(iteration_metrics_list) > 0) {
      iteration_metrics <- do.call(rbind, iteration_metrics_list)
      all_model_metrics[[iteration]] <- iteration_metrics
    }
  }
  
  # Combine and average results
  if (length(all_iterations_results) > 0) {
    # Combine all predictions
    all_predictions_combined <- do.call(rbind, all_iterations_results)
    
    # Combine all model metrics
    all_metrics_combined <- do.call(rbind, all_model_metrics)
    
    # Calculate average predictions by test year, event, and player
    averaged_predictions <- all_predictions_combined %>%
      group_by(Test_Year, EventID, PlayerID) %>%
      summarise(
        # Model Score Statistics
        Model_Score_Mean = mean(Model_Score, na.rm = TRUE),
        Model_Score_Median = median(Model_Score, na.rm = TRUE),
        
        # GLM Odds Probability Statistics
        GLM_Odds_Probability_Mean = mean(GLM_Odds_Probability, na.rm = TRUE),
        GLM_Odds_Probability_Median = median(GLM_Odds_Probability, na.rm = TRUE),
        
        # GLM Implied Probability Statistics
        GLM_ImpliedProb_Probability_Mean = mean(GLM_ImpliedProb_Probability, na.rm = TRUE),
        GLM_ImpliedProb_Probability_Median = median(GLM_ImpliedProb_Probability, na.rm = TRUE),
        
        # Platt Probability Statistics
        Platt_Probability_Mean = mean(Platt_Probability, na.rm = TRUE),
        Platt_Probability_Median = median(Platt_Probability, na.rm = TRUE),
        
        # Static values
        Market_Win = first(Market_Win),
        Position = first(Position),
        Market_Odds = first(Market_Odds),
        
        .groups = 'drop'
      )
    
    return(list(
      market_name = market_name,
      all_predictions = all_predictions_combined,
      averaged_predictions = averaged_predictions,
      model_metrics = all_metrics_combined,
      target_variable = market_config$target_col,
      market_config = market_config
    ))
  }
}

# ===== DATA SETUP =====
cat("\n=== LOADING DATA ===\n")

# Load data - using here() for relative path or provide full path
data_path <- here("Data", "PGA_Processed.xlsx")

# If here() doesn't work, prompt for path
if (!file.exists(data_path)) {
  data_path <- "./Weekly_Modelling/Input/PGA_Processed.xlsx"
}

if (!file.exists(data_path)) {
  stop("Data file not found. Please update data_path variable with correct location.")
}

df_all <- read_excel(data_path)

cat("Initial data rows:", nrow(df_all), "\n")

# Add year information
year_info <- get_year_pairs(df_all)
df_all <- year_info$df_with_year
year_pairs <- year_info$year_pairs

cat("Found", length(year_pairs), "year pairs for training/testing:\n")
for (pair in year_pairs) {
  cat("  Train:", pair$train_year, 
      "-> Test:", pair$test_year, "\n")
}

# Define base model variables (without odds - these will be added per market)
base_model_vars <- c(
  "rating_vs_field_best",
  "rating",
  "rating_vs_field_worst",
  "yr3_All",
  "Top5_rank",
  "compat2",
  "sgp_field_zscore",
  "sgtee_field_zscore",
  "course",
  "sgatg_vs_field_median", 
  "sgapp_vs_field_median",
  "Starts_Not10",
  "current",
  "location", 
  "field"
)

# ===== MAIN PROCESSING LOOP =====
cat("\n=== STARTING MARKET PROCESSING ===\n")
market_results <- list()

for (market_name in names(BETTING_MARKETS)) {
  market_config <- BETTING_MARKETS[[market_name]]
  
  result <- process_market_yearly_validation(
    market_name = market_name,
    market_config = market_config,
    df_all = df_all,
    base_model_vars = base_model_vars,
    year_pairs = year_pairs
  )
  
  market_results[[market_name]] <- result
}

# ===== CALIBRATION ANALYSIS AND RESULTS EXPORT =====
cat("\n=== GENERATING CALIBRATION ANALYSIS ===\n")

# Function to create calibration data for plotting
create_calibration_data <- function(predictions, prob_col, method_name) {
  predictions %>%
    mutate(
      prob_bin = cut(!!sym(prob_col), 
                     breaks = seq(0, 1, 0.01), 
                     include.lowest = TRUE)
    ) %>%
    group_by(prob_bin) %>%
    summarise(
      count = n(),
      predicted_prob = mean(!!sym(prob_col), na.rm = TRUE),
      Market_Win = mean(Market_Win, na.rm = TRUE),
      difference = predicted_prob - Market_Win,
      method = method_name,
      .groups = 'drop'
    ) %>%
    filter(count >= 5)
}

# Calculate calibration metrics function
calc_calibration_metrics <- function(cal_data) {
  list(
    brier_score = mean((cal_data$predicted_prob - cal_data$Market_Win)^2, na.rm = TRUE),
    reliability = mean(abs(cal_data$difference), na.rm = TRUE),
    mean_abs_error = mean(abs(cal_data$difference), na.rm = TRUE),
    weighted_reliability = sum(cal_data$count * abs(cal_data$difference)) / sum(cal_data$count)
  )
}

# Create main results directory
main_results_dir <- paste0("./Results/", MODEL_NAME, "/")
if (!dir.exists(main_results_dir)) {
  dir.create(main_results_dir, recursive = TRUE)
}

# Process results for each market
for (market_name in names(market_results)) {
  result <- market_results[[market_name]]
  
  cat("Processing results for", market_name, "market...\n")
  
  # Create calibration data for all three methods using MEAN values
  cal_data_glm_odds <- create_calibration_data(
    result$averaged_predictions, 
    "GLM_Odds_Probability_Median", 
    "GLM (Odds)"
  )
  cal_data_glm_prob <- create_calibration_data(
    result$averaged_predictions, 
    "GLM_ImpliedProb_Probability_Median", 
    "GLM (Implied Prob)"
  )
  cal_data_platt <- create_calibration_data(
    result$averaged_predictions, 
    "Platt_Probability_Median", 
    "Platt Scaling"
  )
  
  # Combine calibration data
  cal_data_combined <- rbind(cal_data_glm_odds, cal_data_glm_prob, cal_data_platt)
  
  # Calculate calibration metrics
  glm_odds_metrics <- calc_calibration_metrics(cal_data_glm_odds)
  glm_prob_metrics <- calc_calibration_metrics(cal_data_glm_prob)
  platt_metrics <- calc_calibration_metrics(cal_data_platt)
  
  # Create comparison calibration plot
  calibration_plot <- ggplot(cal_data_combined, 
                             aes(x = predicted_prob, y = Market_Win, color = method)) +
    geom_point(aes(size = count), alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    geom_smooth(method = "loess", se = FALSE) +
    facet_wrap(~method) +
    labs(
      x = "Predicted Probability",
      y = paste("Win_Rate"),
      size = "Number of Bets",
      color = "Calibration Method",
      title = paste(market_name, "Market - 1-Year Training")
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Create results directory for this market
  market_results_dir <- paste0(main_results_dir, market_name, "/")
  if (!dir.exists(market_results_dir)) {
    dir.create(market_results_dir, recursive = TRUE)
  }
  
  # Save calibration plot
  ggsave(
    paste0(market_results_dir, market_name, "_calibration_comparison.png"), 
    calibration_plot, 
    width = 12, 
    height = 6, 
    dpi = 300
  )
  
  # Save all individual predictions
  write.csv(
    result$all_predictions, 
    paste0(market_results_dir, market_name, "_All_Iterations_Predictions.csv"), 
    row.names = FALSE
  )
  
  # Save averaged predictions
  write.csv(
    result$averaged_predictions, 
    paste0(market_results_dir, market_name, "_Averaged_Predictions.csv"), 
    row.names = FALSE
  )
  
  # Save model performance metrics
  write.csv(
    result$model_metrics,
    paste0(market_results_dir, market_name, "_Model_Performance_Metrics.csv"),
    row.names = FALSE
  )
  
  # Save calibration comparison data
  write.csv(
    cal_data_combined, 
    paste0(market_results_dir, market_name, "_Calibration_Comparison.csv"), 
    row.names = FALSE
  )
  
  # Save calibration metrics summary
  calibration_metrics_summary <- data.frame(
    Method = c("GLM_Odds", "GLM_ImpliedProb", "Platt_Scaling"),
    Brier_Score = c(glm_odds_metrics$brier_score, 
                    glm_prob_metrics$brier_score, 
                    platt_metrics$brier_score),
    Mean_Absolute_Error = c(glm_odds_metrics$mean_abs_error, 
                            glm_prob_metrics$mean_abs_error, 
                            platt_metrics$mean_abs_error),
    Weighted_Reliability = c(glm_odds_metrics$weighted_reliability, 
                             glm_prob_metrics$weighted_reliability, 
                             platt_metrics$weighted_reliability)
  )
  
  write.csv(
    calibration_metrics_summary, 
    paste0(market_results_dir, market_name, "_Calibration_Metrics.csv"), 
    row.names = FALSE
  )
}

# ===== CROSS-MARKET COMPARISON =====
cat("\n=== CREATING CROSS-MARKET COMPARISON ===\n")

if (length(market_results) > 1) {
  all_metrics <- list()
  
  for (market_name in names(market_results)) {
    result <- market_results[[market_name]]
    
    # Calculate metrics for each method
    cal_data_glm_odds <- create_calibration_data(
      result$averaged_predictions, 
      "GLM_Odds_Probability_Median", 
      "GLM (Odds)"
    )
    cal_data_glm_prob <- create_calibration_data(
      result$averaged_predictions, 
      "GLM_ImpliedProb_Probability_Median", 
      "GLM (Implied Prob)"
    )
    cal_data_platt <- create_calibration_data(
      result$averaged_predictions, 
      "Platt_Probability_Median", 
      "Platt Scaling"
    )
    
    glm_odds_metrics <- calc_calibration_metrics(cal_data_glm_odds)
    glm_prob_metrics <- calc_calibration_metrics(cal_data_glm_prob)
    platt_metrics <- calc_calibration_metrics(cal_data_platt)
    
    # Store metrics
    all_metrics[[market_name]] <- data.frame(
      Market = market_name,
      Method = c("GLM_Odds", "GLM_ImpliedProb", "Platt_Scaling"),
      Brier_Score = c(glm_odds_metrics$brier_score, 
                      glm_prob_metrics$brier_score, 
                      platt_metrics$brier_score),
      Mean_Absolute_Error = c(glm_odds_metrics$mean_abs_error, 
                              glm_prob_metrics$mean_abs_error, 
                              platt_metrics$mean_abs_error),
      Weighted_Reliability = c(glm_odds_metrics$weighted_reliability, 
                               glm_prob_metrics$weighted_reliability, 
                               platt_metrics$weighted_reliability),
      Total_Predictions = nrow(result$averaged_predictions)
    )
  }
  
  # Combine all metrics
  combined_metrics <- do.call(rbind, all_metrics)
  
  # Save combined metrics comparison
  write.csv(
    combined_metrics, 
    paste0(main_results_dir, "All_Markets_Calibration_Comparison.csv"), 
    row.names = FALSE
  )
}

# ===== SUMMARY STATISTICS =====
cat("\n=== GENERATING SUMMARY STATISTICS ===\n")

summary_stats <- list()

for (market_name in names(market_results)) {
  result <- market_results[[market_name]]
  
  # Calculate summary statistics
  summary_stats[[market_name]] <- data.frame(
    Market = market_name,
    Total_Predictions = nrow(result$averaged_predictions),
    Total_Iterations = NUM_ITERATIONS,
    CV_Repetitions = CV_REPETITIONS,
    Unique_Test_Years = length(unique(result$all_predictions$Test_Year)),
    Mean_Model_Score = mean(result$averaged_predictions$Model_Score_Mean, na.rm = TRUE),
    SD_Model_Score = mean(result$averaged_predictions$Model_Score_SD, na.rm = TRUE),
    Training_Window = "1 year"  # Added to indicate training window
  )
}

summary_df <- do.call(rbind, summary_stats)

write.csv(
  summary_df,
  paste0(main_results_dir, "Summary_Statistics.csv"),
  row.names = FALSE
)

# Print timing information
cat("\n=== SCRIPT COMPLETED ===\n")
cat("Script completed at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
end_time <- Sys.time()
total_time <- end_time - start_time
cat("Total runtime:", round(total_time, 2), attr(total_time, "units"), "\n")
cat("\nResults saved to:", main_results_dir, "\n")
cat("\nNOTE: Using 1-year training windows\n")