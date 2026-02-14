# ===== CONFIGURATION =====
setwd("C:/Projects/Golf/Weekly_Modelling")
set.seed(42)
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.7
TRAINING_WINDOW_SIZE <- 80 
NUMBER_OF_RUNS <- 15
MARKETS_TO_RUN <- c("Top10", "Top20")

# Tour configuration - define expected file patterns
TOUR_CONFIG <- list(
  Euro = list(
    name = "European Tour", 
    historical_file = "./Input/Euro_Processed.xlsx",
    weekly_file = "./Input/This_Week_Euro_Processed.xlsx"
  ),
  PGA = list(
    name = "PGA Tour",
    historical_file = "./Input/PGA_Processed.xlsx",
    weekly_file = "./Input/This_Week_PGA_Processed.xlsx"
  )
)

start_time <- Sys.time()

# Define betting markets to process with tour-specific calibration methods
BETTING_MARKETS <- list(
  "Winner" = list(target_col = "win", odds_col = "Win_odds", position_threshold = 1, model_odds_cols = c("Win_odds", "Top5_odds"), market_size = 1),
  "Top5" = list(target_col = "top_5", odds_col = "Top5_odds", position_threshold = 5, model_odds_cols = c("Top5_odds", "Top10_odds"), market_size = 5),
  "Top10" = list(target_col = "top_10", odds_col = "Top10_odds", position_threshold = 10, model_odds_cols = c("Top10_odds", "Top20_odds"), market_size = 10),
  "Top20" = list(target_col = "top_20", odds_col = "Top20_odds", position_threshold = 20, model_odds_cols = c("Top20_odds"), market_size = 20)
)

# Tour-specific calibration method preferences
CALIBRATION_METHODS <- list(
  PGA = list(
    "Winner" = "glm_odds",
    "Top5" = "platt",
    "Top10" = "glm_prob", 
    "Top20" = "platt"
  ),
  Euro = list(
    "Winner" = "glm_odds",
    "Top5" = "glm_prob",
    "Top10" = "glm_prob",
    "Top20" = "platt"
  )
)

# ===== TOURNAMENT DETECTION FUNCTIONS =====
detect_available_tournaments <- function() {
  available_tours <- list()
  
  cat("=== DETECTING AVAILABLE TOURNAMENT FILES ===\n")
  
  for (tour_key in names(TOUR_CONFIG)) {
    tour_info <- TOUR_CONFIG[[tour_key]]
    
    # Check if both historical and weekly files exist
    hist_exists <- file.exists(tour_info$historical_file)
    weekly_exists <- file.exists(tour_info$weekly_file)
    
    cat("\n", tour_info$name, ":\n")
    cat("  Historical:", ifelse(hist_exists, "Found", "Missing"), tour_info$historical_file, "\n")
    cat("  Weekly:", ifelse(weekly_exists, "Found", "Missing"), tour_info$weekly_file, "\n")
    
    # Only include tournament if both files exist
    if (hist_exists && weekly_exists) {
      available_tours[[tour_key]] <- tour_info
      cat("  Status: Ready for modeling\n")
    } else {
      missing_files <- c()
      if (!hist_exists) missing_files <- c(missing_files, "historical")
      if (!weekly_exists) missing_files <- c(missing_files, "weekly")
      cat("  Status: Skipped (missing", paste(missing_files, collapse = " and "), "files)\n")
    }
  }
  
  cat("\n=== SUMMARY ===\n")
  if (length(available_tours) > 0) {
    cat("Available tournaments for modeling:\n")
    for (tour_key in names(available_tours)) {
      cat("  -", available_tours[[tour_key]]$name, "\n")
    }
  } else {
    cat("No tournaments available for modeling - missing required files\n")
  }
  
  return(available_tours)
}

load_tournament_data <- function(tour_info, tour_name) {
  cat("\n--- Loading", tour_name, "Data ---\n")
  
  # Load historical data
  df_historical <- read_excel(tour_info$historical_file)
  df_historical <- df_historical[complete.cases(df_historical),]
  cat("Historical data:", nrow(df_historical), "records loaded\n")
  
  # Load weekly data
  df_upcoming <- read_excel(tour_info$weekly_file)  
  df_upcoming <- df_upcoming[complete.cases(df_upcoming),]
  cat("Weekly data:", nrow(df_upcoming), "players loaded\n")
  
  return(list(
    historical = df_historical,
    upcoming = df_upcoming
  ))
}

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
process_betting_market_single_run <- function(market_name, market_config, train_data, df_upcoming, base_model_vars, run_number, tour_name, tour_key) {
  
  cat("\n--- Run", run_number, "-", market_name, "---\n")
  
  # Get market-specific model variables
  market_model_vars <- get_market_model_vars(market_config, base_model_vars)
  
  # Check available variables for this market
  missing_vars_hist <- market_model_vars[market_model_vars %not in% names(train_data)]
  missing_vars_upcoming <- market_model_vars[market_model_vars %not in% names(df_upcoming)]
  available_market_vars <- market_model_vars[market_model_vars %in% names(train_data) & market_model_vars %in% names(df_upcoming)]

  
  MODEL_NAME <- paste0(tour_name, "_", market_name, "_Run", run_number, "_", format(Sys.time(), "%d%m"))
  
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
  
  # Determine which calibration method to use for this tour/market combination
  selected_method <- CALIBRATION_METHODS[[tour_key]][[market_name]]
  
  # Select the appropriate probability based on tour-specific preferences
  if (selected_method == "glm_odds") {
    final_probability <- predictions$GLM_Odds_Probability
    method_name <- "GLM_Odds"
  } else if (selected_method == "glm_prob") {
    final_probability <- predictions$GLM_Prob_Probability
    method_name <- "GLM_Prob"
  } else if (selected_method == "platt") {
    final_probability <- predictions$Platt_Probability
    method_name <- "Platt"
  }
  
  # Return predictions with selected calibration method
  result_predictions <- data.frame(
    Surname = predictions$surname,
    Firstname = predictions$firstname,
    Rating = predictions$rating,
    Model_Score = predictions$Model_Score,
    Market_Odds = predictions[[market_config$odds_col]],
    Final_Probability = final_probability,
    Calibration_Method = method_name
  ) %>%
    arrange(Market_Odds)
  
  return(result_predictions)
}

# Average results across runs and apply tour-specific calibration methods
process_betting_market_averaged <- function(market_name, market_config, train_data, df_upcoming, base_model_vars, training_events, tour_name, tour_key) {
  
  cat("\n=== PROCESSING", market_name, "MARKET WITH", NUMBER_OF_RUNS, "RUNS ===\n")
  
  # Get the calibration method for this tour/market combination
  selected_method <- CALIBRATION_METHODS[[tour_key]][[market_name]]
  method_display_names <- list(
    "glm_odds" = "GLM with Market Odds",
    "glm_prob" = "GLM with Implied Probability", 
    "platt" = "Platt Scaling"
  )

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
      run_number = run,
      tour_name = tour_name,
      tour_key = tour_key
    )
    
    if (!is.null(result)) {
      run_results[[run]] <- result
    }
  }
  
  # Average the results across runs
  cat("Averaging results across", length(run_results), "runs for", market_name, "\n")
  
  # Start with the first run's structure
  averaged_results <- run_results[[1]]
  
  # Average the relevant columns across all runs
  if (length(run_results) > 1) {
    for (run in 2:length(run_results)) {
      averaged_results$Model_Score <- averaged_results$Model_Score + run_results[[run]]$Model_Score
      averaged_results$Final_Probability <- averaged_results$Final_Probability + run_results[[run]]$Final_Probability
    }
    # Divide by number of runs to get average
    averaged_results$Model_Score <- averaged_results$Model_Score / length(run_results)
    averaged_results$Final_Probability <- averaged_results$Final_Probability / length(run_results)
  }
  
  # Calculate normalised probabilities and odds
  market_size <- market_config$market_size
  total_probability <- sum(averaged_results$Final_Probability)
  
  # Normalize probabilities to sum to market_size
  averaged_results$normalised_Probability <- (averaged_results$Final_Probability / total_probability) * market_size
  normalised_total <- sum(averaged_results$normalised_Probability)
  
  # Calculate model odds for both original and normalised probabilities
  averaged_results$Model_Odds <- 1 / averaged_results$Final_Probability
  averaged_results$normalised_Model_Odds <- 1 / averaged_results$normalised_Probability

  # Rename columns to be market-specific for the main results
  names(averaged_results)[names(averaged_results) == "Market_Odds"] <- paste0(market_name, "_Odds")
  names(averaged_results)[names(averaged_results) == "Model_Score"] <- paste0(market_name, "_Model_Score")
  names(averaged_results)[names(averaged_results) == "Final_Probability"] <- paste0(market_name, "_Probability")
  names(averaged_results)[names(averaged_results) == "normalised_Probability"] <- paste0(market_name, "_normalised_Probability")
  names(averaged_results)[names(averaged_results) == "Model_Odds"] <- paste0(market_name, "_Model_Odds")
  names(averaged_results)[names(averaged_results) == "normalised_Model_Odds"] <- paste0(market_name, "_normalised_Model_Odds")
  names(averaged_results)[names(averaged_results) == "Calibration_Method"] <- paste0(market_name, "_Calibration_Method")
  
  return(list(
    averaged_predictions = averaged_results,
    model_name = paste0(tour_name, "_", market_name, "_", selected_method, "_", NUMBER_OF_RUNS, "runs"),
    market_name = market_name,
    number_of_runs = length(run_results),
    calibration_method = selected_method,
    market_size = market_size,
    probability_sum = normalised_total
  ))
}

# Process single tournament
process_tournament <- function(tour_key, tour_info, base_model_vars) {
  tour_name <- tour_info$name
  
  cat("PROCESSING", toupper(tour_name), "\n")

  # Load tournament data
  tournament_data <- load_tournament_data(tour_info, tour_name)
  df_historical <- tournament_data$historical
  df_upcoming <- tournament_data$upcoming
  
  # Get training data from most recent events  
  training_result <- get_training_data(df_historical, TRAINING_WINDOW_SIZE, base_model_vars)
  train_data <- training_result$train_data
  training_events <- training_result$training_events
  
  # Validate market selection
  available_markets <- names(BETTING_MARKETS)
  SELECTED_BETTING_MARKETS <- BETTING_MARKETS[MARKETS_TO_RUN]
  
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
      training_events = training_events,
      tour_name = tour_key,
      tour_key = tour_key
    )
    
    if (!is.null(result)) {
      all_averaged_predictions[[market_name]] <- result$averaged_predictions
      market_results[[market_name]] <- result
    }
  }
  
  # Export results for this tournament
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
    
    # Save tournament-specific results
    tournament_file <- paste0(results_dir, tour_key, "_Weekly_Predictions_", format(Sys.time(), "%d%m"), ".xlsx")
    
    # Create workbook with multiple sheets
    wb <- createWorkbook()
    
    # Individual market sheets with all calibration methods
    for (market_name in names(all_averaged_predictions)) {
      addWorksheet(wb, paste0(market_name, "_Market"))
      writeData(wb, paste0(market_name, "_Market"), all_averaged_predictions[[market_name]])
    }
    
    saveWorkbook(wb, tournament_file, overwrite = TRUE)
    
    cat("\n", tour_name, "results saved to:", tournament_file, "\n")
    
    return(list(
      tour_name = tour_name,
      predictions = all_averaged_predictions,
      output_file = tournament_file,
      field_size = nrow(df_upcoming)
    ))
  } else {
    cat("\nNo valid predictions generated for", tour_name, "\n")
    return(NULL)
  }
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

# ===== MAIN EXECUTION =====
cat("=== GOLF TOURNAMENT MODELING - MULTI-TOUR VERSION ===\n")
cat("Current working directory:", getwd(), "\n")

# Step 1: Detect available tournaments
available_tournaments <- detect_available_tournaments()

# Step 2: Exit if no tournaments available
if (length(available_tournaments) == 0) {
  cat("\n No tournaments available for modeling. Please ensure you have both historical and weekly files for at least one tour.\n")
  cat("Expected files:\n")
  for (tour_key in names(TOUR_CONFIG)) {
    tour_info <- TOUR_CONFIG[[tour_key]]
    cat("  ", tour_info$name, ":\n")
    cat("    -", tour_info$historical_file, "\n")
    cat("    -", tour_info$weekly_file, "\n")
  }
  stop("Modeling cannot proceed without tournament data")
}

# Step 3: Define base model variables
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

# Step 4: Process each available tournament
tournament_results <- list()

for (tour_key in names(available_tournaments)) {
  tour_info <- available_tournaments[[tour_key]]
  
  tryCatch({
    result <- process_tournament(tour_key, tour_info, base_model_vars)
    if (!is.null(result)) {
      tournament_results[[tour_key]] <- result
    }
  }, error = function(e) {
    cat("\n Error processing", tour_info$name, ":", e$message, "\n")
    cat("Continuing with other tournaments...\n")
  })
}

# Step 5: Final summary
end_time <- Sys.time()
total_time <- round(difftime(end_time, start_time, units = "hours"), 2)

cat("MODELING COMPLETE - FINAL SUMMARY\n")

if (length(tournament_results) > 0) {
  cat("Successfully processed tournaments:\n")
  
  for (tour_key in names(tournament_results)) {
    result <- tournament_results[[tour_key]]
    cat("\n", result$tour_name, ":\n")
    cat("Field Size:", result$field_size, "players\n")
    cat("Markets:", length(result$predictions), "processed\n")
    cat("Output:", basename(result$output_file), "\n")
  }
  
  cat("\n All prediction files saved to ./Output/ directory\n")
  cat("Models used:", paste(MODELS_TO_PROCESS, collapse = ", "), "\n")
  cat("Runs per market:", NUMBER_OF_RUNS, "\n")
  cat("Calibration methods by tour:\n")
  cat("PGA: Winner=GLM_Odds, Top5=Platt, Top10=GLM_Prob, Top20=Platt\n")
  cat("Euro: Winner=GLM_Odds, Top5=GLM_Prob, Top10=GLM_Prob, Top20=Platt\n")
  
} else {
  cat("No tournaments were successfully processed\n")
  cat("Please check your data files and try again\n")
}

cat("\n️ Total processing time:", total_time, "hours\n")