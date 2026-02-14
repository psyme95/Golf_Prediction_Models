# ===== CONFIGURATION SECTION =====
# Set working directory and parameters
setwd("C:/Projects/Golf")
set.seed(42)

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.7
MODEL_NAME <- paste0("MultiMarket_Rolling_", format(Sys.time(), "%d%m_%H%M"))
MIN_PLAYERS_PER_EVENT <- 1
start_time <- Sys.time()

# Rolling window parameters
TRAINING_WINDOW_SIZE <- 80  # Number of events to train on
RETRAIN_FREQUENCY <- 40      # Retrain model every k events
NUM_ITERATIONS <- 20       # Number of times to repeat the process

# Undersampling parameters
UNDERSAMPLE_TRAINING <- FALSE
PLAYERS_PER_EVENT <- 100  # Total players per event in training (winner + losers)

# Define betting markets to test
BETTING_MARKETS <- list(
  "Winner" = list(
    target_col = "win",
    odds_col = "Win_odds",
    position_threshold = 1,
    model_odds_cols = c("Win_odds", "Top5_odds")  # Winner + next level
  ),
  "Top5" = list(
    target_col = "top_5",
    odds_col = "Top5_odds", 
    position_threshold = 5,
    model_odds_cols = c("Top5_odds", "Top10_odds")  # Top5 + next level
  ),
  "Top10" = list(
    target_col = "top_10",
    odds_col = "Top10_odds",
    position_threshold = 10,
    model_odds_cols = c("Top10_odds", "Top20_odds")  # Top10 + next level
  ),
  "Top20" = list(
    target_col = "top_20", 
    odds_col = "Top20_odds",
    position_threshold = 20,
    model_odds_cols = c("Top20_odds", "Top40_odds")  # Top20 + next level
  )
)

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
  
  return(list(A = result$par[1], B = result$par[2]))
}

# Apply Platt scaling to new scores using fitted parameters
predict_platt_scaling <- function(scores, platt_params) {
  1 / (1 + exp(platt_params$A * scores + platt_params$B))
}

# Model training function (adapted for any target variable)
train_ensemble_model <- function(train_data, model_vars, target_variable, log_file = NULL) {
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
  
  return(myBiomodEM)
}

# Function to get market-specific model variables
get_market_model_vars <- function(market_config, base_model_vars) {
  # Start with base variables (excluding all odds columns)
  base_vars_no_odds <- base_model_vars[!base_model_vars %in% c("Win_odds", "Top5_odds", "Top10_odds", "Top20_odds", "Top40_odds")]
  
  # Add market-specific odds columns
  market_specific_vars <- c(base_vars_no_odds, market_config$model_odds_cols)
  
  return(market_specific_vars)
}

# Function to get rolling training data with undersampling (adapted for any target)
get_rolling_training_data <- function(df_all, training_size, available_vars, 
                                      current_event_number, target_variable,
                                      iteration = 1,
                                      total_players_per_event = 50) {
  unique_events <- df_all %>%
    select(eventID, event_number) %>%
    distinct() %>%
    arrange(event_number)
  
  # Calculate training window end point (event before current test event)
  training_end_event <- current_event_number - 1
  training_start_event <- max(1, training_end_event - training_size + 1)
  
  # Get training events (rolling window)
  training_events <- training_start_event:training_end_event
  training_events <- training_events[training_events > 0]  # Ensure positive event numbers
  
  cat("    Event", current_event_number, ": Training on events", min(training_events), "to", max(training_events), 
      "(", length(training_events), "events)\n")
  
  # Extract raw training data (before undersampling)
  essential_cols <- c("playerID", "eventID", "event_number", "posn", "win", "top_5", "top_10", "top_20", "top_40",
                      "Win_odds", "Top5_odds", "Top10_odds", "Top20_odds", "Top40_odds")
  
  # Only include columns that exist
  available_essential_cols <- essential_cols[essential_cols %in% names(df_all)]
  all_required_cols <- c(available_essential_cols, available_vars)
  all_required_cols <- unique(all_required_cols)
  
  train_data_raw <- df_all %>%
    filter(event_number %in% training_events) %>%
    select(all_of(all_required_cols))
  
  # Apply undersampling ONLY to training data
  if (UNDERSAMPLE_TRAINING && nrow(train_data_raw) > 0) {
    set.seed(42 + iteration * 1000 + current_event_number * 100)  # Ensure reproducible undersampling
    
    train_data_undersampled <- train_data_raw %>%
      group_by(eventID) %>%
      group_modify(~ {
        # Get winners and losers for this event based on target variable
        winners <- .x[.x[[target_variable]] == 1, ]
        losers <- .x[.x[[target_variable]] == 0, ]
        
        # Always keep ALL winners
        n_winners <- nrow(winners)
        
        # Calculate how many losers to sample
        n_losers_to_sample <- total_players_per_event - n_winners
        n_losers_available <- nrow(losers)
        
        if (n_losers_to_sample > 0 && n_losers_available > 0) {
          # Don't sample more losers than available
          n_losers_final <- min(n_losers_to_sample, n_losers_available)
          
          # Randomly sample losers
          if (n_losers_final >= n_losers_available) {
            # Take all losers if we need more than available
            sampled_losers <- losers
          } else {
            # Randomly sample the required number of losers
            sampled_indices <- sample(nrow(losers), n_losers_final, replace = FALSE)
            sampled_losers <- losers[sampled_indices, ]
          }
          
          # Combine winners and sampled losers
          result <- rbind(winners, sampled_losers)
        } else {
          # If no losers needed/available, just return winners
          result <- winners
        }
        
        return(result)
      }) %>%
      ungroup()
    
    train_data <- train_data_undersampled
  } else {
    train_data <- train_data_raw
  }
  
  return(list(
    train_data = train_data,
    training_events = training_events,
    training_start = training_start_event,
    training_end = training_end_event
  ))
}

# Process single betting market with rolling window validation
process_market_rolling_validation <- function(market_name, market_config, df_all, base_model_vars) {
  
  cat("\n=== PROCESSING", market_name, "MARKET ===\n")
  
  # Check if target variable exists
  if (!market_config$target_col %in% names(df_all)) {
    cat("Warning: Target variable", market_config$target_col, "not found in data. Skipping", market_name, "\n")
    return(NULL)
  }
  
  # Get market-specific model variables
  market_model_vars <- get_market_model_vars(market_config, base_model_vars)
  
  # Check available variables
  missing_vars <- market_model_vars[market_model_vars %not in% names(df_all)]
  available_vars <- market_model_vars[market_model_vars %in% names(df_all)]
  
  cat("Available variables for", market_name, ":", length(available_vars), "/", length(market_model_vars), "\n")
  if (length(missing_vars) > 0) {
    cat("Missing variables:", paste(missing_vars, collapse = ", "), "\n")
  }
  
  if (length(available_vars) < 5) {
    cat("Insufficient variables for", market_name, ". Need at least 5, have", length(available_vars), "\n")
    return(NULL)
  }
  
  # Initialize storage for all iterations
  all_iterations_results <- list()
  
  cat("Starting", NUM_ITERATIONS, "iterations of rolling window validation for", market_name, "...\n")
  cat("Training window size:", TRAINING_WINDOW_SIZE, "events\n")
  cat("Model retrain frequency: every", RETRAIN_FREQUENCY, "events\n")
  if (UNDERSAMPLE_TRAINING) {
    cat("Training undersampling enabled:", PLAYERS_PER_EVENT, "players per event\n")
  }
  cat("\n")
  
  for (iteration in 1:NUM_ITERATIONS) {
    cat("=== ITERATION", iteration, "OF", NUM_ITERATIONS, "FOR", market_name, "===\n")
    
    # Get all available events for this iteration
    unique_events <- df_all %>%
      select(eventID, event_number) %>%
      distinct() %>%
      arrange(event_number)
    
    total_events <- max(unique_events$event_number)
    
    # Start testing from event where we have sufficient training data
    first_test_event <- TRAINING_WINDOW_SIZE + 1
    
    if (first_test_event > total_events) {
      cat("Not enough events for iteration", iteration, ". Skipping.\n")
      next
    }
    
    # Initialize variables for model caching
    current_model <- NULL
    current_calibration_models <- NULL
    last_training_event <- 0
    iteration_predictions_list <- list()
    
    # Loop through test events
    for (test_event in first_test_event:total_events) {
      
      # Check if we need to retrain the model
      events_since_training <- test_event - 1 - last_training_event
      
      if (is.null(current_model) || events_since_training >= RETRAIN_FREQUENCY) {
        
        cat("  Retraining model for event", test_event, "(", events_since_training, "events since last training)\n")
        
        # Get rolling training data
        training_setup <- get_rolling_training_data(df_all, TRAINING_WINDOW_SIZE, available_vars, 
                                                    test_event, market_config$target_col, iteration, PLAYERS_PER_EVENT)
        
        train_data <- training_setup$train_data
        
        if (nrow(train_data) == 0) {
          cat("    No training data available for event", test_event, ". Skipping.\n")
          next
        }
        
        # Train ensemble model with market-specific target
        current_model <- train_ensemble_model(train_data, available_vars, market_config$target_col)
        
        # Get training predictions for calibration fitting
        training_scores <- current_model@models.prediction@val$pred
        training_labels <- train_data[[market_config$target_col]]
        training_odds <- train_data[[market_config$odds_col]]
        
        # Train ALL THREE calibration models using training data
        # Calculate market implied probability
        training_implied_prob <- 1 / training_odds
        
        # 1. Logistic Regression Calibration (with odds)
        logit_model_data_odds <- data.frame(
          EventID = train_data$eventID,
          PlayerID = train_data$playerID,
          Model_Score = training_scores,
          Actual_Result = training_labels,
          Market_Odds = training_odds
        )
        
        calibration_model_glm_odds <- glm(Actual_Result ~ Model_Score + Market_Odds,
                                          data = logit_model_data_odds,
                                          family = binomial())
        
        # 2. Logistic Regression Calibration (with implied probability)
        logit_model_data_prob <- data.frame(
          EventID = train_data$eventID,
          PlayerID = train_data$playerID,
          Model_Score = training_scores,
          Actual_Result = training_labels,
          Implied_Prob = training_implied_prob
        )
        
        calibration_model_glm_prob <- glm(Actual_Result ~ Model_Score + Implied_Prob,
                                          data = logit_model_data_prob,
                                          family = binomial())
        
        # 3. Platt Scaling Calibration
        platt_params <- apply_platt_scaling(training_scores, training_labels)
        
        # Store calibration models
        current_calibration_models <- list(
          glm_odds = calibration_model_glm_odds,
          glm_prob = calibration_model_glm_prob,
          platt = platt_params
        )
        
        last_training_event <- test_event - 1
        cat("    Training completed with", nrow(train_data), "observations\n")
      }
      
      # Make predictions on current test event
      test_columns <- c("playerID", "eventID", "event_number", "posn", market_config$target_col, "Quality", 
                        market_config$odds_col, "rating", "Lay_odds", "Lay_top5", "Lay_top10", "Lay_top20", available_vars)
      test_columns <- unique(test_columns)
      available_test_columns <- test_columns[test_columns %in% names(df_all)]
      
      current_test_data <- df_all %>%
        filter(event_number == test_event) %>%
        select(all_of(available_test_columns))
      
      if (nrow(current_test_data) == 0) {
        cat("    No test data available for event", test_event, "\n")
        next
      }
      
      # Prepare data for biomod projection
      test_expl <- current_test_data[, available_vars]
      test_resp.xy <- current_test_data[, c("playerID", "eventID")]
      
      # Run biomod projection on current test event
      myBiomodProj <- BIOMOD_EnsembleForecasting(
        bm.em = current_model,
        bm.proj = NULL,
        proj.name = paste0("iter", iteration, "_event", test_event, "_", market_name),
        new.env = test_expl,
        new.env.xy = test_resp.xy,
        models.chosen = "all",
        metric.binary = "TSS",
        metric.filter = "TSS",
        na.rm = TRUE
      )
      
      # Process predictions for current test event
      test_predictions <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
      colnames(test_predictions)[3] <- "Model_Score"
      
      # Merge with test data
      essential_cols <- c("eventID", "playerID", "event_number", "posn", market_config$target_col)
      optional_cols <- c(market_config$odds_col, "rating", "Quality", "Lay_odds", "Lay_top5", "Lay_top10", "Lay_top20")
      merge_cols <- intersect(names(current_test_data), c(essential_cols, optional_cols))
      
      test_predictions <- merge(
        test_predictions,
        current_test_data[, merge_cols],
        by = c("eventID", "playerID"), all.x = TRUE
      )
      
      # Apply ALL THREE calibration methods
      # Calculate market implied probability for test data
      test_implied_prob <- 1 / test_predictions[[market_config$odds_col]]
      
      # 1. GLM Calibration (with odds)
      calibration_data_glm_odds <- data.frame(
        Model_Score = test_predictions$Model_Score,
        Market_Odds = test_predictions[[market_config$odds_col]]
      )
      
      test_predictions$GLM_Odds_Probability <- predict(current_calibration_models$glm_odds, 
                                                       calibration_data_glm_odds, 
                                                       type = "response")
      
      # 2. GLM Calibration (with implied probability)
      calibration_data_glm_prob <- data.frame(
        Model_Score = test_predictions$Model_Score,
        Implied_Prob = test_implied_prob
      )
      
      test_predictions$GLM_ImpliedProb_Probability <- predict(current_calibration_models$glm_prob, 
                                                              calibration_data_glm_prob, 
                                                              type = "response")
      
      # 3. Platt Scaling Calibration
      test_predictions$Platt_Probability <- predict_platt_scaling(test_predictions$Model_Score, current_calibration_models$platt)
      
      # Store predictions for this event
      event_predictions <- data.frame(
        Iteration = iteration,
        EventNumber = test_predictions$event_number,
        EventID = test_predictions$eventID,
        PlayerID = test_predictions$playerID,
        Model_Score = round(test_predictions$Model_Score, 4),
        Actual_Result = test_predictions[[market_config$target_col]],
        Actual_Position = test_predictions$posn,
        Market_Odds = test_predictions[[market_config$odds_col]],
        Lay_odds = if("Lay_odds" %in% names(test_predictions)) test_predictions$Lay_odds else NA,
        GLM_Odds_Probability = test_predictions$GLM_Odds_Probability,
        GLM_ImpliedProb_Probability = test_predictions$GLM_ImpliedProb_Probability,
        Platt_Probability = test_predictions$Platt_Probability,
        Retrained_For_Event = (events_since_training >= RETRAIN_FREQUENCY || is.null(current_model))
      )
      
      iteration_predictions_list[[length(iteration_predictions_list) + 1]] <- event_predictions
    }
    
    # Combine all predictions for this iteration
    if (length(iteration_predictions_list) > 0) {
      iteration_predictions <- do.call(rbind, iteration_predictions_list)
      all_iterations_results[[iteration]] <- iteration_predictions
      
      retrains <- sum(iteration_predictions$Retrained_For_Event)
      total_events_tested <- length(unique(iteration_predictions$EventNumber))
      
      cat("Iteration", iteration, "completed for", market_name, ":", nrow(iteration_predictions), "predictions made\n")
      cat("  Events tested:", total_events_tested, "\n")
      cat("  Model retrains:", retrains, "\n")
      cat("  Retrain frequency: every ~", round(total_events_tested / max(1, retrains), 1), "events\n")
    } else {
      cat("Iteration", iteration, "completed for", market_name, ": No predictions made\n")
    }
    
    cat("\n")
  }
  
  # Combine and average results for this market
  if (length(all_iterations_results) > 0) {
    cat("=== COMBINING RESULTS FOR", market_name, "===\n")
    
    # Combine all predictions
    all_predictions_combined <- do.call(rbind, all_iterations_results)
    
    # Calculate average predictions by event and player
    cat("Calculating average predictions across iterations for", market_name, "...\n")
    
    averaged_predictions <- all_predictions_combined %>%
      group_by(EventNumber, EventID, PlayerID) %>%
      summarise(
        Model_Score = mean(Model_Score, na.rm = TRUE),
        Actual_Result = first(Actual_Result),
        Actual_Position = first(Actual_Position),
        Market_Odds = first(Market_Odds),
        Lay_odds = first(Lay_odds),
        GLM_Odds_Probability = mean(GLM_Odds_Probability, na.rm = TRUE),
        GLM_ImpliedProb_Probability = mean(GLM_ImpliedProb_Probability, na.rm = TRUE),
        Platt_Probability = mean(Platt_Probability, na.rm = TRUE),
        Times_Retrained = sum(Retrained_For_Event, na.rm = TRUE),
        .groups = 'drop'
      )
    
    return(list(
      market_name = market_name,
      all_predictions = all_predictions_combined,
      averaged_predictions = averaged_predictions,
      target_variable = market_config$target_col,
      market_config = market_config
    ))
  } else {
    cat("No results for", market_name, "\n")
    return(NULL)
  }
}

# ===== LOAD REQUIRED PACKAGES =====
library(biomod2)
library(dplyr)
library(readxl)
library(lubridate)
library(ggplot2)

# Define 'not in' operator
`%not in%` <- Negate(`%in%`)

# ===== DATA SETUP =====
# Load and validate data
df_all <- read_excel("./Data/PGA_250725_Processed.xlsx")
df_all <- df_all[complete.cases(df_all),]

# Create sequential event numbering based on dates
df_all <- df_all %>%
  mutate(Date = Date) %>%
  arrange(Date) %>%
  group_by(eventID) %>%
  mutate(event_date = first(Date)) %>%
  ungroup() %>%
  arrange(event_date) %>%
  mutate(event_number = as.numeric(as.factor(eventID)))

# Get unique events in chronological order
unique_events <- df_all %>%
  select(eventID, event_number, event_date) %>%
  distinct() %>%
  arrange(event_number)

total_events <- max(unique_events$event_number)

# Define base model variables (without odds - these will be added per market)
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

# ===== MAIN PROCESSING LOOP =====
# Process each betting market independently
market_results <- list()

for (market_name in names(BETTING_MARKETS)) {
  market_config <- BETTING_MARKETS[[market_name]]
  
  result <- process_market_rolling_validation(
    market_name = market_name,
    market_config = market_config,
    df_all = df_all,
    base_model_vars = base_model_vars
  )
  
  if (!is.null(result)) {
    market_results[[market_name]] <- result
    cat("\n", market_name, "market processing completed successfully\n")
  } else {
    cat("\n", market_name, "market processing failed\n")
  }
}

# ===== CALIBRATION ANALYSIS AND RESULTS EXPORT =====
# Function to create calibration data for plotting (define outside the loop)
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
      actual_rate = mean(Actual_Result, na.rm = TRUE),
      difference = predicted_prob - actual_rate,
      method = method_name,
      .groups = 'drop'
    ) %>%
    filter(count >= 5)  # Only bins with sufficient data
}

# Calculate calibration metrics function (define outside the loop)
calc_calibration_metrics <- function(cal_data) {
  list(
    brier_score = mean((cal_data$predicted_prob - cal_data$actual_rate)^2, na.rm = TRUE),
    reliability = mean(abs(cal_data$difference), na.rm = TRUE),
    mean_abs_error = mean(abs(cal_data$difference), na.rm = TRUE),
    weighted_reliability = sum(cal_data$count * abs(cal_data$difference)) / sum(cal_data$count)
  )
}

# Process results for each market
for (market_name in names(market_results)) {
  result <- market_results[[market_name]]
  
  cat("\n=== ANALYZING CALIBRATION FOR", market_name, "===\n")
  
  # Create calibration data for all three methods
  cal_data_glm_odds <- create_calibration_data(result$averaged_predictions, "GLM_Odds_Probability", "GLM (Odds)")
  cal_data_glm_prob <- create_calibration_data(result$averaged_predictions, "GLM_ImpliedProb_Probability", "GLM (Implied Prob)")
  cal_data_platt <- create_calibration_data(result$averaged_predictions, "Platt_Probability", "Platt Scaling")
  
  # Combine calibration data
  cal_data_combined <- rbind(cal_data_glm_odds, cal_data_glm_prob, cal_data_platt)
  
  # Calculate calibration metrics
  glm_odds_metrics <- calc_calibration_metrics(cal_data_glm_odds)
  glm_prob_metrics <- calc_calibration_metrics(cal_data_glm_prob)
  platt_metrics <- calc_calibration_metrics(cal_data_platt)
  
  cat(market_name, "GLM (Odds) Calibration Metrics:\n")
  cat("  Brier Score:", round(glm_odds_metrics$brier_score, 4), "\n")
  cat("  Mean Absolute Error:", round(glm_odds_metrics$mean_abs_error, 4), "\n")
  cat("  Weighted Reliability:", round(glm_odds_metrics$weighted_reliability, 4), "\n")
  
  cat("\n", market_name, "GLM (Implied Probability) Calibration Metrics:\n")
  cat("  Brier Score:", round(glm_prob_metrics$brier_score, 4), "\n")
  cat("  Mean Absolute Error:", round(glm_prob_metrics$mean_abs_error, 4), "\n")
  cat("  Weighted Reliability:", round(glm_prob_metrics$weighted_reliability, 4), "\n")
  
  cat("\n", market_name, "Platt Scaling Metrics:\n")
  cat("  Brier Score:", round(platt_metrics$brier_score, 4), "\n")
  cat("  Mean Absolute Error:", round(platt_metrics$mean_abs_error, 4), "\n")
  cat("  Weighted Reliability:", round(platt_metrics$weighted_reliability, 4), "\n")
  
  # Create comparison calibration plot
  calibration_plot <- ggplot(cal_data_combined, aes(x = predicted_prob, y = actual_rate, color = method)) +
    geom_point(aes(size = count), alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
    geom_smooth(method = "loess", se = FALSE) +
    facet_wrap(~method) +
    labs(x = "Predicted Probability",
         y = paste("Actual", market_name, "Rate"),
         size = "Number of Bets",
         color = "Calibration Method",
         title = paste("Rolling Window Calibration Comparison for", market_name, "Market")) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Create results directory for this market
  market_results_dir <- paste0("./Results/", MODEL_NAME, "/", market_name, "/")
  if (!dir.exists(market_results_dir)) {
    dir.create(market_results_dir, recursive = TRUE)
  }
  
  # Save calibration plot
  ggsave(paste0(market_results_dir, market_name, "_calibration_comparison.png"), 
         calibration_plot, width = 12, height = 6, dpi = 300)
  
  # Save all individual predictions for this market
  write.csv(result$all_predictions, 
            paste0(market_results_dir, market_name, "_All_Iterations_Predictions.csv"), 
            row.names = FALSE)
  
  # Save averaged predictions for this market
  write.csv(result$averaged_predictions, 
            paste0(market_results_dir, market_name, "_Averaged_Predictions.csv"), 
            row.names = FALSE)
  
  # Save calibration comparison data for this market
  write.csv(cal_data_combined, 
            paste0(market_results_dir, market_name, "_Calibration_Comparison.csv"), 
            row.names = FALSE)
  
  # Save calibration metrics for this market
  calibration_metrics_summary <- data.frame(
    Method = c("GLM_Odds", "GLM_ImpliedProb", "Platt_Scaling"),
    Brier_Score = c(glm_odds_metrics$brier_score, glm_prob_metrics$brier_score, platt_metrics$brier_score),
    Mean_Absolute_Error = c(glm_odds_metrics$mean_abs_error, glm_prob_metrics$mean_abs_error, platt_metrics$mean_abs_error),
    Weighted_Reliability = c(glm_odds_metrics$weighted_reliability, glm_prob_metrics$weighted_reliability, platt_metrics$weighted_reliability)
  )
  
  write.csv(calibration_metrics_summary, 
            paste0(market_results_dir, market_name, "_Calibration_Metrics.csv"), 
            row.names = FALSE)
  
  # Create and save retraining analysis for this market
  retrain_analysis <- result$all_predictions %>%
    group_by(Iteration, EventNumber) %>%
    summarise(
      Retrained = any(Retrained_For_Event, na.rm = TRUE),
      Players_Predicted = n(),
      .groups = 'drop'
    ) %>%
    arrange(Iteration, EventNumber)
  
  write.csv(retrain_analysis, 
            paste0(market_results_dir, market_name, "_Retraining_Analysis.csv"), 
            row.names = FALSE)
  
  cat("\n", market_name, "results saved to:", market_results_dir, "\n")
}

# ===== CROSS-MARKET COMPARISON =====
cat("\n=== CROSS-MARKET COMPARISON ===\n")

# Create combined calibration metrics comparison
if (length(market_results) > 1) {
  all_metrics <- list()
  
  for (market_name in names(market_results)) {
    result <- market_results[[market_name]]
    
    # Calculate metrics for each method
    cal_data_glm_odds <- create_calibration_data(result$averaged_predictions, "GLM_Odds_Probability", "GLM (Odds)")
    cal_data_glm_prob <- create_calibration_data(result$averaged_predictions, "GLM_ImpliedProb_Probability", "GLM (Implied Prob)")
    cal_data_platt <- create_calibration_data(result$averaged_predictions, "Platt_Probability", "Platt Scaling")
    
    glm_odds_metrics <- calc_calibration_metrics(cal_data_glm_odds)
    glm_prob_metrics <- calc_calibration_metrics(cal_data_glm_prob)
    platt_metrics <- calc_calibration_metrics(cal_data_platt)
    
    # Store metrics
    all_metrics[[market_name]] <- data.frame(
      Market = market_name,
      Method = c("GLM_Odds", "GLM_ImpliedProb", "Platt_Scaling"),
      Brier_Score = c(glm_odds_metrics$brier_score, glm_prob_metrics$brier_score, platt_metrics$brier_score),
      Mean_Absolute_Error = c(glm_odds_metrics$mean_abs_error, glm_prob_metrics$mean_abs_error, platt_metrics$mean_abs_error),
      Weighted_Reliability = c(glm_odds_metrics$weighted_reliability, glm_prob_metrics$weighted_reliability, platt_metrics$weighted_reliability),
      Total_Predictions = nrow(result$averaged_predictions)
    )
  }
  
  # Combine all metrics
  combined_metrics <- do.call(rbind, all_metrics)
  
  # Save combined metrics comparison
  main_results_dir <- paste0("./Results/", MODEL_NAME, "/")
  write.csv(combined_metrics, 
            paste0(main_results_dir, "All_Markets_Calibration_Comparison.csv"), 
            row.names = FALSE)
  
  # Create summary statistics
  summary_stats <- combined_metrics %>%
    group_by(Market) %>%
    summarise(
      Best_Brier_Score = min(Brier_Score, na.rm = TRUE),
      Best_MAE = min(Mean_Absolute_Error, na.rm = TRUE),
      Best_Reliability = min(Weighted_Reliability, na.rm = TRUE),
      Best_Method_Brier = Method[which.min(Brier_Score)],
      Best_Method_MAE = Method[which.min(Mean_Absolute_Error)],
      Best_Method_Reliability = Method[which.min(Weighted_Reliability)],
      Total_Predictions = first(Total_Predictions),
      .groups = 'drop'
    )
  
  write.csv(summary_stats, 
            paste0(main_results_dir, "Market_Performance_Summary.csv"), 
            row.names = FALSE)
  
  # Print summary
  cat("\nMARKET PERFORMANCE SUMMARY:\n")
  for (i in 1:nrow(summary_stats)) {
    market <- summary_stats$Market[i]
    cat("\n", market, "Market:\n")
    cat("  Best Brier Score:", round(summary_stats$Best_Brier_Score[i], 4), 
        "(", summary_stats$Best_Method_Brier[i], ")\n")
    cat("  Best MAE:", round(summary_stats$Best_MAE[i], 4), 
        "(", summary_stats$Best_Method_MAE[i], ")\n")
    cat("  Best Reliability:", round(summary_stats$Best_Reliability[i], 4), 
        "(", summary_stats$Best_Method_Reliability[i], ")\n")
    cat("  Total Predictions:", summary_stats$Total_Predictions[i], "\n")
  }
}

# ===== FINAL SUMMARY =====
cat("\n=== MULTI-MARKET ANALYSIS COMPLETE ===\n")
cat("Markets processed:", paste(names(market_results), collapse = ", "), "\n")
cat("Iterations per market:", NUM_ITERATIONS, "\n")
cat("Training window size:", TRAINING_WINDOW_SIZE, "events\n")
cat("Retrain frequency: every", RETRAIN_FREQUENCY, "events\n")
cat("Results structure:\n")
cat("  - Main directory: ./Results/", MODEL_NAME, "/\n")
cat("  - Individual market directories for each processed market\n")
cat("  - Cross-market comparison files in main directory\n")

cat("\nKey outputs per market:\n")
cat("  - [Market]_All_Iterations_Predictions.csv: Individual iteration results\n")
cat("  - [Market]_Averaged_Predictions.csv: Final averaged predictions with all three calibration methods\n")
cat("  - [Market]_Calibration_Comparison.csv: Calibration analysis data\n")
cat("  - [Market]_Calibration_Metrics.csv: Summary metrics comparing all three methods\n")
cat("  - [Market]_Retraining_Analysis.csv: Details of when models were retrained\n")
cat("  - [Market]_calibration_comparison.png: Visual comparison of calibration quality\n")

cat("\nCross-market comparison files:\n")
cat("  - All_Markets_Calibration_Comparison.csv: Combined metrics for all markets\n")
cat("  - Market_Performance_Summary.csv: Best performing methods per market\n")

cat("\nSUMMARY:\n")
cat("  Approach: Multi-market rolling window with periodic retraining\n")
cat("  Markets tested:", length(market_results), "out of", length(BETTING_MARKETS), "configured\n")
cat("  Calibration methods: GLM (Odds), GLM (Implied Prob), Platt Scaling for each market\n")
if (length(market_results) > 0) {
  total_predictions <- sum(sapply(market_results, function(x) nrow(x$averaged_predictions)))
  cat("  Total predictions across all markets:", total_predictions, "\n")
}

# Print timing information
cat("\nScript completed at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
end_time <- Sys.time()

end_time - start_time