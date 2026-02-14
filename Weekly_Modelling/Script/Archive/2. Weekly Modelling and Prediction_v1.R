# ===== CONFIGURATION =====
setwd("C:/Projects/Golf/Weekly_Modelling")
set.seed(42)
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.7
TRAINING_WINDOW_SIZE <- 80 
NUMBER_OF_RUNS <- 50
MARKETS_TO_RUN <- c("Winner", "Top5", "Top10", "Top20")
HISTORICAL_DATA_FILE <- "./Input/PGA_Processed.xlsx"
UPCOMING_EVENT_FILE <- "./Input/This_Week_Processed.xlsx"
start_time <- Sys.time()

# Define betting markets to process
BETTING_MARKETS <- list(
  "Winner" = list(target_col = "win", odds_col = "Win_odds", position_threshold = 1, model_odds_cols = c("Win_odds", "Top5_odds")),
  "Top5" = list(target_col = "top_5", odds_col = "Top5_odds", position_threshold = 5, model_odds_cols = c("Top5_odds", "Top10_odds")),
  "Top10" = list(target_col = "top_10", odds_col = "Top10_odds", position_threshold = 10, model_odds_cols = c("Top10_odds", "Top20_odds")),
  "Top20" = list(target_col = "top_20", odds_col = "Top20_odds", position_threshold = 20, model_odds_cols = c("Top20_odds"))
)

# ===== FUNCTIONS =====
# Platt Scaling functions
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
  
  return(list(A = result$par[1], B = result$par[2]))
}

# Apply Platt scaling to new scores using fitted parameters
predict_platt_scaling <- function(scores, platt_params) {
  1 / (1 + exp(platt_params$A * scores + platt_params$B))
}

# Enhanced model training function that returns both ensemble model and calibration models
train_ensemble_model_with_calibration <- function(train_data, model_vars, target_variable, market_config) {
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
  
  # Biomod options (same as original)
  user.rf <- list('_allData_allRun' = list(ntree=1500,
                                           nodesize=350,
                                           mtry = floor(sqrt(length(model_vars)))
  ))
  user.XGBOOST <- list('_allData_allRun' = list(nrounds = 50,
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
  
  # Get training predictions for calibration fitting
  training_scores <- myBiomodEM@models.prediction@val$pred
  training_labels <- train_data[[target_variable]]
  training_odds <- train_data[[market_config$odds_col]]
  training_implied_prob <- 1 / training_odds
  
  # Fit all three calibration methods
  cat("    Fitting calibration models...\n")
  
  # 1. GLM with odds
  calibration_model_glm_odds <- glm(training_labels ~ training_scores + training_odds,
                                    family = binomial())
  
  # 2. GLM with implied probability
  calibration_model_glm_prob <- glm(training_labels ~ training_scores + training_implied_prob,
                                    family = binomial())
  
  # 3. Platt scaling
  platt_params <- apply_platt_scaling(training_scores, training_labels)
  
  calibration_models <- list(
    glm_odds = calibration_model_glm_odds,
    glm_prob = calibration_model_glm_prob,
    platt = platt_params
  )
  
  cat("    Calibration models fitted successfully\n")
  
  return(list(
    ensemble_model = myBiomodEM,
    calibration_models = calibration_models
  ))
}

# Function to get the most recent N events for training
get_training_data <- function(df_historical, window_size, available_vars) {
  # Get unique events sorted by date
  unique_events <- df_historical %>%
    select(eventID, Date) %>%
    distinct() %>%
    arrange(Date)
  
  # Get the most recent N events
  total_events <- nrow(unique_events)
  recent_events <- tail(unique_events, window_size)
  
  cat("Training on", nrow(recent_events), "most recent events out of", total_events, "total events\n")
  cat("Date range:", min(recent_events$Date), "to", max(recent_events$Date), "\n")
  
  # Extract training data for these events
  train_data <- df_historical %>%
    filter(eventID %in% recent_events$eventID) %>%
    select(playerID, eventID, posn, win, top_5, top_10, top_20, top_40,
           Win_odds, Top5_odds, Top10_odds, Top20_odds,
           all_of(available_vars))
  
  return(list(train_data = train_data, training_events = recent_events))
}

# Function to get market-specific model variables
get_market_model_vars <- function(market_config, base_model_vars) {
  # Start with base variables (excluding odds columns)
  base_vars_no_odds <- base_model_vars[!base_model_vars %in% c("Win_odds", "Top5_odds", "Top10_odds", "Top20_odds")]
  
  # Add market-specific odds columns
  market_specific_vars <- c(base_vars_no_odds, market_config$model_odds_cols)
  
  return(market_specific_vars)
}

# Process single betting market for one run
process_betting_market_single_run <- function(market_name, market_config, train_data, df_upcoming, base_model_vars, run_number) {
  
  cat("\n--- Run", run_number, "-", market_name, "---\n")
  
  # Get market-specific model variables
  market_model_vars <- get_market_model_vars(market_config, base_model_vars)
  
  # Check available variables for this market
  missing_vars_hist <- market_model_vars[market_model_vars %not in% names(train_data)]
  missing_vars_upcoming <- market_model_vars[market_model_vars %not in% names(df_upcoming)]
  available_market_vars <- market_model_vars[market_model_vars %in% names(train_data) & market_model_vars %in% names(df_upcoming)]
  
  # Check if target variable exists in training data
  if (!market_config$target_col %in% names(train_data)) {
    cat("Warning: Target variable", market_config$target_col, "not found in training data. Skipping", market_name, "\n")
    return(NULL)
  }
  
  # Check if primary odds column exists in prediction data  
  if (!market_config$odds_col %in% names(df_upcoming)) {
    cat("Warning: Odds column", market_config$odds_col, "not found in prediction data. Skipping", market_name, "\n")
    return(NULL)
  }
  
  # Check if we have enough variables to proceed
  if (length(available_market_vars) < 5) {
    cat("Warning: Insufficient variables available for", market_name, "market. Need at least 5, have", length(available_market_vars), "\n")
    return(NULL)
  }
  
  MODEL_NAME <- paste0(market_name, "_Run", run_number, "_", format(Sys.time(), "%d%m"))
  
  # Train the ensemble model with calibration models
  trained_result <- train_ensemble_model_with_calibration(train_data, available_market_vars, market_config$target_col, market_config)
  
  # Make predictions on upcoming event
  test_columns <- c("surname", "firstname", market_config$odds_col, "rating", available_market_vars)
  test_columns <- unique(test_columns)
  available_test_columns <- test_columns[test_columns %in% names(df_upcoming)]
  
  prediction_data <- df_upcoming %>%
    select(all_of(available_test_columns))
  
  # Prepare data for biomod prediction
  test_expl <- prediction_data[, available_market_vars]
  test_resp.xy <- prediction_data[, c("surname", "firstname")]
  
  # Run biomod projection
  myBiomodProj <- BIOMOD_EnsembleForecasting(
    bm.em = trained_result$ensemble_model,
    bm.proj = NULL,
    proj.name = MODEL_NAME,
    new.env = test_expl,
    new.env.xy = test_resp.xy,
    models.chosen = "all",
    metric.binary = "TSS",
    metric.filter = "TSS",
    na.rm = TRUE
  )
  
  # Process predictions
  predictions <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
  colnames(predictions)[3] <- "Model_Score"
  
  # Merge with original data
  essential_cols <- c("surname", "firstname", market_config$odds_col, "rating")
  merge_cols <- intersect(names(prediction_data), essential_cols)
  
  predictions <- merge(
    predictions,
    prediction_data[, merge_cols],
    by = c("surname", "firstname"), all.x = TRUE
  )
  
  # Apply all three calibration methods
  # Prepare calibration data
  calibration_data_odds <- data.frame(
    training_scores = predictions$Model_Score,
    training_odds = predictions[[market_config$odds_col]]
  )
  
  calibration_data_prob <- data.frame(
    training_scores = predictions$Model_Score,
    training_implied_prob = 1 / predictions[[market_config$odds_col]]
  )
  
  # Apply calibrations
  predictions$GLM_Odds_Probability <- predict(trained_result$calibration_models$glm_odds, 
                                              calibration_data_odds, type = "response")
  predictions$GLM_Prob_Probability <- predict(trained_result$calibration_models$glm_prob, 
                                              calibration_data_prob, type = "response")
  predictions$Platt_Probability <- predict_platt_scaling(predictions$Model_Score, 
                                                         trained_result$calibration_models$platt)
  
  # Return predictions with all calibration methods
  result_predictions <- data.frame(
    Surname = predictions$surname,
    Firstname = predictions$firstname,
    Rating = predictions$rating,
    Model_Score = predictions$Model_Score,
    Market_Odds = predictions[[market_config$odds_col]],
    GLM_Odds_Probability = predictions$GLM_Odds_Probability,
    GLM_Prob_Probability = predictions$GLM_Prob_Probability,
    Platt_Probability = predictions$Platt_Probability
  ) %>%
    arrange(Market_Odds)
  
  return(result_predictions)
}

# Average results across runs and apply all calibration methods
process_betting_market_averaged <- function(market_name, market_config, train_data, df_upcoming, base_model_vars, training_events) {
  
  cat("\n=== PROCESSING", market_name, "MARKET WITH", NUMBER_OF_RUNS, "RUNS ===\n")
  
  # Store results from each run
  run_results <- list()
  
  # Run the model multiple times
  for (run in 1:NUMBER_OF_RUNS) {
    result <- process_betting_market_single_run(
      market_name = market_name,
      market_config = market_config,
      train_data = train_data,
      df_upcoming = df_upcoming,
      base_model_vars = base_model_vars,
      run_number = run
    )
    
    if (!is.null(result)) {
      run_results[[run]] <- result
    }
  }
  
  # Check if we have valid results
  if (length(run_results) == 0) {
    cat("No valid results for", market_name, "\n")
    return(NULL)
  }
  
  # Average the results across runs
  cat("Averaging results across", length(run_results), "runs for", market_name, "\n")
  
  # Start with the first run's structure
  averaged_results <- run_results[[1]]
  
  # Average the relevant columns across all runs
  if (length(run_results) > 1) {
    for (run in 2:length(run_results)) {
      averaged_results$Model_Score <- averaged_results$Model_Score + run_results[[run]]$Model_Score
      averaged_results$GLM_Odds_Probability <- averaged_results$GLM_Odds_Probability + run_results[[run]]$GLM_Odds_Probability
      averaged_results$GLM_Prob_Probability <- averaged_results$GLM_Prob_Probability + run_results[[run]]$GLM_Prob_Probability
      averaged_results$Platt_Probability <- averaged_results$Platt_Probability + run_results[[run]]$Platt_Probability
    }
    # Divide by number of runs to get average
    averaged_results$Model_Score <- averaged_results$Model_Score / length(run_results)
    averaged_results$GLM_Odds_Probability <- averaged_results$GLM_Odds_Probability / length(run_results)
    averaged_results$GLM_Prob_Probability <- averaged_results$GLM_Prob_Probability / length(run_results)
    averaged_results$Platt_Probability <- averaged_results$Platt_Probability / length(run_results)
  }
  
  # Calculate model odds for all three calibration methods
  averaged_results$GLM_Odds_ModelOdds <- round(1 / averaged_results$GLM_Odds_Probability, 2)
  averaged_results$GLM_Prob_ModelOdds <- round(1 / averaged_results$GLM_Prob_Probability, 2)
  averaged_results$Platt_ModelOdds <- round(1 / averaged_results$Platt_Probability, 2)
  
  # Round the results
  averaged_results$Model_Score <- round(averaged_results$Model_Score, 4)
  averaged_results$GLM_Odds_Probability <- round(averaged_results$GLM_Odds_Probability, 4)
  averaged_results$GLM_Prob_Probability <- round(averaged_results$GLM_Prob_Probability, 4)
  averaged_results$Platt_Probability <- round(averaged_results$Platt_Probability, 4)
  
  # Rename columns to be market-specific
  names(averaged_results)[names(averaged_results) == "Market_Odds"] <- paste0(market_name, "_Odds")
  names(averaged_results)[names(averaged_results) == "Model_Score"] <- paste0(market_name, "_Model_Score")
  names(averaged_results)[names(averaged_results) == "GLM_Odds_Probability"] <- paste0(market_name, "_GLM_Odds_Probability")
  names(averaged_results)[names(averaged_results) == "GLM_Prob_Probability"] <- paste0(market_name, "_GLM_Prob_Probability")
  names(averaged_results)[names(averaged_results) == "Platt_Probability"] <- paste0(market_name, "_Platt_Probability")
  names(averaged_results)[names(averaged_results) == "GLM_Odds_ModelOdds"] <- paste0(market_name, "_GLM_Odds_ModelOdds")
  names(averaged_results)[names(averaged_results) == "GLM_Prob_ModelOdds"] <- paste0(market_name, "_GLM_Prob_ModelOdds")
  names(averaged_results)[names(averaged_results) == "Platt_ModelOdds"] <- paste0(market_name, "_Platt_ModelOdds")
  
  return(list(
    averaged_predictions = averaged_results,
    model_name = paste0(market_name, "_Averaged_", NUMBER_OF_RUNS, "runs"),
    market_name = market_name,
    number_of_runs = length(run_results)
  ))
}

# ===== PACKAGES =====
library(biomod2)
library(dplyr)
library(readxl)
library(openxlsx)
library(lubridate)
library(ggplot2)

# Define 'not in' operator
`%not in%` <- Negate(`%in%`)

# ===== DATA LOADING =====
# Load historical events data
df_historical <- read_excel(HISTORICAL_DATA_FILE)
df_historical <- df_historical[complete.cases(df_historical),]

# Load upcoming event data
df_upcoming <- read_excel(UPCOMING_EVENT_FILE)
df_upcoming <- df_upcoming[complete.cases(df_upcoming),]

# ===== VALIDATE MARKET SELECTION =====
# Check that specified markets exist
available_markets <- names(BETTING_MARKETS)
invalid_markets <- MARKETS_TO_RUN[!MARKETS_TO_RUN %in% available_markets]

# Filter markets to only those specified
SELECTED_BETTING_MARKETS <- BETTING_MARKETS[MARKETS_TO_RUN]

# ===== VARIABLE SETUP =====
# Define base model variables (without odds - these will be added dynamically)
base_model_vars <- c("rating_vs_field_best",
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
                     "field")

# ===== MODEL TRAINING AND PREDICTION =====
# Get training data from most recent events  
training_result <- get_training_data(df_historical, TRAINING_WINDOW_SIZE, base_model_vars)
train_data <- training_result$train_data
training_events <- training_result$training_events

# Process all selected betting markets with multiple runs and all calibration methods
all_averaged_predictions <- list()
market_results <- list()

for (market_name in names(SELECTED_BETTING_MARKETS)) {
  market_config <- SELECTED_BETTING_MARKETS[[market_name]]
  
  result <- process_betting_market_averaged(
    market_name = market_name,
    market_config = market_config, 
    train_data = train_data,
    df_upcoming = df_upcoming,
    base_model_vars = base_model_vars,
    training_events = training_events
  )
  
  if (!is.null(result)) {
    all_averaged_predictions[[market_name]] <- result$averaged_predictions
    market_results[[market_name]] <- result
    
    cat("\n", market_name, "predictions completed with", result$number_of_runs, "runs\n")
    cat("Field Size:", nrow(result$averaged_predictions), "\n")
    cat("All three calibration methods applied\n")
  }
}

# ===== COMBINE RESULTS AND EXPORT =====
if (length(all_averaged_predictions) > 0) {
  
  # Create combined dataset with all averaged markets
  base_data <- all_averaged_predictions[[1]][, c("Surname", "Firstname", "Rating")]
  
  for (market_name in names(all_averaged_predictions)) {
    market_data <- all_averaged_predictions[[market_name]]
    market_cols <- names(market_data)[!names(market_data) %in% c("Surname", "Firstname", "Rating")]
    
    base_data <- merge(base_data, market_data[, c("Surname", "Firstname", market_cols)],
                       by = c("Surname", "Firstname"), all = TRUE)
  }
  
  # Create results directory
  results_dir <- paste0("./Output/")
  if (!dir.exists(results_dir)) {
    dir.create(results_dir, recursive = TRUE)
  }
  
  # Save combined results
  combined_file <- paste0(results_dir, "Weekly_Predictions_", format(Sys.time(), "%d%m"), ".xlsx")
  
  # Create workbook with multiple sheets
  wb <- createWorkbook()
  
  # Individual market sheets with all calibration methods
  for (market_name in names(all_averaged_predictions)) {
    addWorksheet(wb, paste0(market_name, "_Market"))
    writeData(wb, paste0(market_name, "_Market"), all_averaged_predictions[[market_name]])
  }
  
  saveWorkbook(wb, combined_file, overwrite = TRUE)
}

end_time <- Sys.time()
cat("\nOutput saved to:", combined_file,
    "\nTotal processing time:", round(difftime(end_time, start_time, units = "hours"), 2), "hours\n")