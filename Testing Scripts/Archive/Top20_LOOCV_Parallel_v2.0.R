#### Enhanced LOOCV with Sequential Betting Simulation ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
library(foreach)
library(doSNOW)
library(caret)
library(readxl)
library(tidyr)

'%not in%' <- function(x,table) is.na(match(x,table,nomatch=NA_integer_))

# ===== CONFIGURATION SECTION =====
STARTING_BANKROLL <- 1000
FIXED_STAKE_AMOUNT <- 10
MIN_PLAYERS_PER_EVENT <- 120

# Model parameters (from Script 2)
MODELS_TO_PROCESS <- c('GAM', 'GLM', 'RF', 'FDA', 'CTA', 'ANN', 'GBM', 'MAXNET', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.75

# ===== HELPER FUNCTIONS =====
log_message <- function(message, log_file = NULL, console = TRUE, timestamp = TRUE) {
  if (timestamp) {
    full_message <- paste0(Sys.time(), " - ", message, "\n")
  } else {
    full_message <- paste0(message, "\n")
  }
  
  if (console) cat(full_message)
  if (!is.null(log_file)) cat(full_message, file = log_file, append = TRUE)
}

# Script 2's feature engineering (run once at start)
create_derived_features <- function(df, log_file = NULL) {
  log_message("Creating derived features...", log_file)
  
  # Create top 20 target variable
  df$top_20 <- ifelse(df$posn <= 20, 1, 0)
  
  # Event-level features processing
  events <- split(df, df$eventID)
  events <- events[sapply(events, function(df) nrow(df) > 80)]
  df_with_diffs <- data.frame()
  
  log_message(paste("Processing", length(events), "events for relative features..."), log_file)
  
  for (i in seq_along(events)) {
    if (i %% 10 == 0) {
      log_message(paste("Processed", i, "of", length(events), "events"), log_file, console = FALSE)
    }
    
    event_data <- events[[i]]
    
    # Calculate event statistics
    event_mean <- mean(event_data$rating, na.rm = TRUE)
    event_median <- median(event_data$rating, na.rm = TRUE)
    event_max <- max(event_data$rating, na.rm = TRUE)
    event_min <- min(event_data$rating, na.rm = TRUE)
    
    # Create relative features
    event_data$diff_from_mean <- event_data$rating - event_mean
    event_data$diff_from_median <- event_data$rating - event_median
    event_data$diff_from_max <- event_data$rating - event_max
    event_data$diff_from_min <- event_data$rating - event_min
    event_data$rating_normal <- as.numeric(scale(event_data$rating))
    event_data$log_rating <- as.numeric(log(event_data$rating + 50))
    
    df_with_diffs <- rbind(df_with_diffs, event_data)
  }
  
  log_message("Derived features creation completed", log_file)
  return(df_with_diffs)
}

# Script 2's strategy generation
generate_betting_strategies <- function(size_increments = c(10, 20, 30, 40), log_file = NULL) {
  log_message("Generating betting strategies...", log_file)
  
  ranges_df <- data.frame()
  
  for (size in size_increments) {
    max_start <- 81 - size
    for (start_val in seq(1, max_start, by = 10)) {
      end_val <- start_val + size - 1
      
      if (start_val <= end_val && start_val >= 1 && end_val <= 80) {
        ranges_df <- rbind(ranges_df, data.frame(start = start_val, end = end_val, size = size))
      }
    }
  }
  
  ranges_df <- unique(ranges_df)
  strategies <- list()
  
  for (i in 1:nrow(ranges_df)) {
    start_val <- ranges_df$start[i]
    end_val <- ranges_df$end[i]
    strategy_name <- paste0("R", start_val, "to", end_val)
    strategies[[strategy_name]] <- list(
      start = start_val, 
      end = end_val,
      size = end_val - start_val + 1,
      description = paste("Bet on players ranked", start_val, "to", end_val, "by model")
    )
  }
  
  log_message(paste("Generated", length(strategies), "betting strategies"), log_file)
  return(strategies)
}

# Adapted model training (Script 2's approach for LOOCV)
train_ensemble_model_loocv <- function(train_data, available_vars, log_file = NULL) {
  tryCatch({
    resp <- train_data[, "top_20"]
    expl <- train_data[, available_vars]
    
    # Clear any existing biomod2 temporary files
    temp_dir <- tempdir()
    biomod_files <- list.files(temp_dir, pattern = "BIOMOD", full.names = TRUE)
    if (length(biomod_files) > 0) {
      try(file.remove(biomod_files), silent = TRUE)
    }
    
    myBiomodData <- BIOMOD_FormatingData(
      resp.var = resp,
      expl.var = expl,
      resp.name = paste0("PGA_", Sys.getpid(), "_", sample(10000, 1)),  # Unique name per worker
      dir.name = paste0(getwd(), "/Models")
    )
    
    # Create dummy coordinates
    dummy_coords <- data.frame(
      x = 1:length(myBiomodData@data.species),
      y = 1:length(myBiomodData@data.species)
    )
    myBiomodData@coord <- dummy_coords
    
    # Train individual models with more robust settings
    myBiomodModelOut <- BIOMOD_Modeling(
      bm.format = myBiomodData,
      models = MODELS_TO_PROCESS,
      OPT.strategy = "bigboss",
      CV.nb.rep = CV_REPETITIONS,
      CV.perc = CV_PERCENTAGE,
      weights = NULL,
      var.import = 0,  # Disable variable importance to avoid serialization issues
      metric.eval = c('TSS'),
      nb.cpu = 1,
      do.progress = FALSE
      )
    
    # Evaluate models
    myBiomodModelEval <- get_evaluations(myBiomodModelOut)
    
    # Check if evaluation was successful
    if (is.null(myBiomodModelEval) || nrow(myBiomodModelEval) == 0) {
      return(NULL)
    }
    
    allRun.model.names <- myBiomodModelEval[
      which(myBiomodModelEval$run == "allRun" & myBiomodModelEval$metric.eval == "TSS"),
      "full.name"
    ]
    
    if (length(allRun.model.names) == 0) {
      return(NULL)
    }
    
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
    
    # Create ensemble model with error handling
    myBiomodEM <- BIOMOD_EnsembleModeling(
      bm.mod = myBiomodModelOut,
      models.chosen = names(df.metrics),
      em.algo = c('EMwmean'),
      metric.select = c("user.defined"),
      metric.select.thresh = c(0.3),
      metric.select.table = df.metrics,
      metric.eval = c('TSS'),
      var.import = 0,  # Disable variable importance
      EMwmean.decay = 'proportional'
    )
    
    return(myBiomodEM)
    
  }, error = function(e) {
    # Enhanced error logging
    error_msg <- paste("Error in model training:", e$message)
    if (!is.null(log_file)) {
      tryCatch({
        cat(paste0(Sys.time(), " - ", error_msg, "\n"), file = log_file, append = TRUE)
      }, error = function(e2) {})
    }
    return(NULL)
  })
}

# Generate predictions for single event
generate_predictions_loocv <- function(ensemble_model, test_data, available_vars, test_event_id) {
  test_expl <- test_data[, available_vars]
  test_resp.xy <- test_data[, c("playerID", "eventID")]
  
  myBiomodProj <- BIOMOD_EnsembleForecasting(
    bm.em = ensemble_model,
    bm.proj = NULL,
    proj.name = paste0("event_", test_event_id),
    new.env = test_expl,
    new.env.xy = test_resp.xy,
    models.chosen = "all",
    metric.binary = "TSS",
    metric.filter = "TSS",
    na.rm = TRUE
  )
  
  PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
  colnames(PGA_Prediction)[3] <- "Model_Score"
  
  PGA_Prediction <- merge(
    PGA_Prediction,
    test_data[, c("eventID", "playerID", "Top20_Odds", "top20_winnings", "rating", "posn", "top_20")],
    by = c("eventID", "playerID"), all.x = TRUE
  )
  
  PGA_Prediction$Valid_Odds <- PGA_Prediction$Top20_Odds > 1
  
  return(PGA_Prediction)
}

# Calculate betting results for strategy
calculate_strategy_bets <- function(event_predictions, strategy, fixed_stake_amount) {
  # Rank players by model score (lowercase column name)
  event_predictions <- event_predictions %>%
    arrange(desc(model_score), desc(rating)) %>%
    mutate(rank_order = row_number())
  
  # Apply strategy range
  bet_players <- event_predictions %>%
    filter(rank_order >= strategy$start, 
           rank_order <= strategy$end,
           valid_odds == TRUE)
  
  if(nrow(bet_players) == 0) {
    return(list(num_bets = 0, total_stake = 0, total_winnings = 0, profit_loss = 0))
  }
  
  # Fixed stakes
  total_stake <- nrow(bet_players) * fixed_stake_amount
  total_winnings <- sum(ifelse(bet_players$actual_top20 == 1, 
                               bet_players$top20_winnings, 0))
  profit_loss <- total_winnings - total_stake
  
  return(list(
    num_bets = nrow(bet_players),
    total_stake = total_stake,
    total_winnings = total_winnings,
    profit_loss = profit_loss
  ))
}

# ===== MAIN EXECUTION =====
# Create log file
log_file <- paste0("C:/Projects/Golf/temp_status/loocv_sequential_simulation_", format(Sys.time(), "%m%d_%H%M"), ".log")
log_message("Starting LOOCV with Sequential Betting Simulation", log_file)

# Load and prepare data
setwd("C:/Projects/Golf")
df <- read_excel("./Data/PGA_revised_odds.xlsx")
names(df) <- gsub("^_", "X_", names(df))
df$rating <- as.numeric(df$rating)
#df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
#df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df$top40_winnings <- df$Top40_Profit + 10
df$Top20_Winnings <- df$Top20_Profit + 10
df <- df[complete.cases(df),]
df$eventID <- as.numeric(df$eventID)

eventQuality <- read.csv("./Data/FieldQuality.csv")
df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID"))

# Create derived features (run once at start)
df <- create_derived_features(df, log_file)

# Define model variables
model_vars <- c("field", "current_rank", "compat_rank", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "sgapp", "Starts_Not10", "Top20_Odds", "diff_from_max", "current_top20", "compat2", 
                "AvPosn_Rank", "yr3_All_rank", "performance_trend")


available_vars <- model_vars[model_vars %in% names(df)]
log_message(paste("Available model variables:", length(available_vars), "of", length(model_vars)), log_file)

# Create model dataset
model_data <- df %>%
  select(playerID, eventID, posn, top_20, top20_winnings, rating, all_of(available_vars))

# Get unique events for LOOCV
unique_events <- sample(unique(model_data$eventID))
log_message(paste("Starting LOOCV with", length(unique_events), "events"), log_file)

# ===== PHASE 1: LOOCV PREDICTION GENERATION (PARALLEL) =====
# Set up parallel processing (Script 1's approach)
n_cores <- 12
cl <- makeCluster(n_cores, type = "SOCK")
registerDoSNOW(cl)

log_message(paste("Setting up parallel processing with", n_cores, "cores"), log_file)

# Enhanced progress tracking with comprehensive logging
completed_events <- 0
start_time_overall <- Sys.time()
successful_folds <- 0
failed_folds <- 0
skipped_folds <- 0

# Enhanced progress function with detailed logging
progress <- function(n) {
  completed_events <<- completed_events + 1
  setTxtProgressBar(pb, completed_events)
  
  elapsed_time <- as.numeric(difftime(Sys.time(), start_time_overall, units = "mins"))
  if (completed_events > 1) {
    avg_time_per_event <- elapsed_time / completed_events
    remaining_events <- length(unique_events) - completed_events
    eta_minutes <- avg_time_per_event * remaining_events
    eta_time <- Sys.time() + (eta_minutes * 60)
    
    # Console output only (brief)
    cat(paste0(Sys.time(), " - Phase 1 Progress: ", completed_events, "/", length(unique_events), 
               " events completed | Elapsed: ", round(elapsed_time, 1), 
               " mins | ETA: ", format(eta_time, "%H:%M:%S"), "\n"))
  }
  
  flush.console()
}

pb <- txtProgressBar(min = 0, max = length(unique_events), style = 3)
opts <- list(progress = progress)

# Export additional logging variables to workers
clusterExport(cl, c("model_data", "available_vars", "MODELS_TO_PROCESS", 
                    "CV_REPETITIONS", "CV_PERCENTAGE", "MIN_PLAYERS_PER_EVENT",
                    "train_ensemble_model_loocv", "generate_predictions_loocv", "log_file",
                    "unique_events"))

clusterEvalQ(cl, {
  library(biomod2)
  library(dplyr)
  library(data.table)
  library(rpart)
  
  # Helper function for worker logging
  worker_log <- function(message, event_id = NULL, fold_num = NULL) {
    tryCatch({
      if (!is.null(event_id) && !is.null(fold_num)) {
        full_message <- paste0(Sys.time(), " - ", message, "\n")
      } else {
        full_message <- paste0(Sys.time(), " - Worker: ", message, "\n")
      }
      cat(full_message, file = log_file, append = TRUE)
    }, error = function(e) {
      # Ignore logging errors to prevent worker crashes
    })
  }
})

log_message("Starting Phase 1: LOOCV Prediction Generation with enhanced logging", log_file)
log_message(paste("Total events to process:", length(unique_events)), log_file)
log_message(paste("Events range:", min(unique_events), "to", max(unique_events)), log_file)
log_message(paste("Parallel workers:", n_cores), log_file)
log_message(paste("Minimum players per event threshold:", MIN_PLAYERS_PER_EVENT), log_file)
cat("Starting Phase 1: LOOCV Prediction Generation...\n")

# Enhanced LOOCV parallel loop with comprehensive worker logging
loocv_results <- foreach(i = 1:length(unique_events), 
                         .packages = c('biomod2', 'dplyr', 'data.table', 'rpart'),
                         .options.snow = opts,
                         .errorhandling = 'pass',
                         .verbose = FALSE) %dopar% {
                           
                           test_event <- unique_events[i]
                           train_events <- unique_events[-i]
                           worker_start_time <- Sys.time()
                           
                           # Log worker start
                           worker_log(paste("STARTED Event", i, "of", length(unique_events), "- EventID", test_event), test_event, i)
                           
                           # Wrap everything in a single tryCatch to ensure we always return something
                           tryCatch({
                             # Split data
                             train_data <- model_data[model_data$eventID %in% train_events, ]
                             test_data <- model_data[model_data$eventID == test_event, ]
                             
                             # Check data sufficiency
                             if (nrow(test_data) < MIN_PLAYERS_PER_EVENT) {
                               processing_time <- as.numeric(difftime(Sys.time(), worker_start_time, units = "secs"))
                               worker_log(paste("SKIPPED Event", i, "of", length(unique_events), "- EventID", test_event, "- insufficient data (", nrow(test_data), "players)"), test_event, i)
                               return(data.frame(status = "skipped", eventID = test_event, reason = "insufficient_data", stringsAsFactors = FALSE))
                             }
                             
                             # Train model
                             ensemble_model <- train_ensemble_model_loocv(train_data, available_vars, log_file)
                             
                             if (is.null(ensemble_model)) {
                               processing_time <- as.numeric(difftime(Sys.time(), worker_start_time, units = "secs"))
                               worker_log(paste("FAILED Event", i, "of", length(unique_events), "- EventID", test_event, "- model training failed"), test_event, i)
                               return(data.frame(status = "failed", eventID = test_event, reason = "model_training_failed", stringsAsFactors = FALSE))
                             }
                             
                             # Generate predictions
                             predictions <- generate_predictions_loocv(ensemble_model, test_data, available_vars, test_event)
                             
                             # Validate predictions
                             if (is.null(predictions) || nrow(predictions) == 0) {
                               processing_time <- as.numeric(difftime(Sys.time(), worker_start_time, units = "secs"))
                               worker_log(paste("FAILED Event", i, "of", length(unique_events), "- EventID", test_event, "- prediction generation failed"), test_event, i)
                               return(data.frame(status = "failed", eventID = test_event, reason = "prediction_failed", stringsAsFactors = FALSE))
                             }
                             
                             # Return structured predictions
                             result_data <- data.frame(
                               eventID = test_event,
                               playerID = predictions$playerID,
                               model_score = predictions$Model_Score,
                               actual_top20 = predictions$top_20,
                               actual_position = predictions$posn,
                               top20_odds = predictions$Top20_Odds,
                               top20_winnings = predictions$top20_winnings,
                               rating = predictions$rating,
                               valid_odds = predictions$Valid_Odds,
                               fold_number = i,
                               stringsAsFactors = FALSE
                             )
                             
                             processing_time <- as.numeric(difftime(Sys.time(), worker_start_time, units = "secs"))
                             worker_log(paste("COMPLETED Event", i, "of", length(unique_events), "- EventID", test_event, "- generated", nrow(predictions), "predictions in", round(processing_time, 1), "seconds"), test_event, i)
                             
                             return(result_data)
                             
                           }, error = function(e) {
                             processing_time <- as.numeric(difftime(Sys.time(), worker_start_time, units = "secs"))
                             worker_log(paste("ERROR Event", i, "of", length(unique_events), "- EventID", test_event, "-", e$message), test_event, i)
                             return(data.frame(status = "error", eventID = test_event, reason = paste("Error:", e$message), stringsAsFactors = FALSE))
                           })
                         }

close(pb)
stopCluster(cl)

# Enhanced post-processing logging
log_message("Phase 1 parallel processing completed", log_file)

# Process the list of results
successful_predictions <- data.frame()
status_records <- data.frame()

if (length(loocv_results) > 0) {
  log_message(paste("Processing", length(loocv_results), "worker results"), log_file)
  
  for (i in 1:length(loocv_results)) {
    result <- loocv_results[[i]]
    
    if (is.data.frame(result) && nrow(result) > 0) {
      if ("status" %in% names(result)) {
        status_records <- rbind(status_records, result)
      } else {
        successful_predictions <- rbind(successful_predictions, result)
      }
    }
  }
} else {
  log_message("No worker results to process", log_file)
}

# Summary logging
log_message(paste("Found", nrow(successful_predictions), "prediction rows from", length(unique(successful_predictions$eventID)), "events"), log_file)
log_message(paste("Found", nrow(status_records), "status records"), log_file)

if (nrow(status_records) > 0) {
  status_summary <- table(status_records$status)
  for (status_type in names(status_summary)) {
    log_message(paste("Events with status '", status_type, "':", status_summary[status_type]), log_file)
  }
}

if (nrow(successful_predictions) > 0) {
  processed_events <- unique(successful_predictions$eventID)
  successful_folds <- length(processed_events)
  
  log_message(paste("Successful predictions generated for", successful_folds, "events"), log_file)
  log_message(paste("Total predictions:", nrow(successful_predictions)), log_file)
  log_message(paste("Average predictions per event:", round(nrow(successful_predictions) / successful_folds, 1)), log_file)
  
  # Log prediction quality metrics
  score_summary <- summary(successful_predictions$model_score)
  log_message(paste("Model score distribution - Min:", round(score_summary[1], 3), 
                    "Q1:", round(score_summary[2], 3),
                    "Median:", round(score_summary[3], 3),
                    "Q3:", round(score_summary[5], 3),
                    "Max:", round(score_summary[6], 3)), log_file)
  
  valid_odds_pct <- round(sum(successful_predictions$valid_odds) / nrow(successful_predictions) * 100, 1)
  top20_rate <- round(sum(successful_predictions$actual_top20) / nrow(successful_predictions) * 100, 1)
  
  log_message(paste("Valid odds percentage:", valid_odds_pct, "%"), log_file)
  log_message(paste("Actual top 20 rate:", top20_rate, "%"), log_file)
  
  # Event size analysis
  event_sizes <- successful_predictions %>% 
    group_by(eventID) %>% 
    summarise(n_players = n(), .groups = "drop") %>%
    pull(n_players)
  
  log_message(paste("Event sizes - Min:", min(event_sizes), 
                    "Max:", max(event_sizes),
                    "Mean:", round(mean(event_sizes), 1)), log_file)
}

if (nrow(status_records) > 0) {
  status_summary <- table(status_records$status)
  for (status_type in names(status_summary)) {
    log_message(paste("Events with status '", status_type, "':", status_summary[status_type]), log_file)
  }
  
  # Log details of failed/skipped events
  if (any(status_records$status %in% c("failed", "error", "skipped"))) {
    log_message("Details of problematic events:", log_file)
    problem_events <- status_records[status_records$status %in% c("failed", "error", "skipped"), ]
    for (i in 1:nrow(problem_events)) {
      log_message(paste("  Event", problem_events$eventID[i], ":", problem_events$status[i], 
                        "-", problem_events$reason[i]), log_file, timestamp = FALSE)
    }
  }
}

# Final processing time summary
total_processing_time <- as.numeric(difftime(Sys.time(), start_time_overall, units = "mins"))
log_message(paste("Total Phase 1 processing time:", round(total_processing_time, 2), "minutes"), log_file)
log_message(paste("Average time per event:", round(total_processing_time / length(unique_events), 2), "minutes"), log_file)

# Set up for Phase 2
if (nrow(successful_predictions) > 0) {
  loocv_results <- successful_predictions  # Use only successful predictions for Phase 2
  log_message(paste("Proceeding to Phase 2 with", nrow(loocv_results), "predictions from", 
                    length(unique(loocv_results$eventID)), "events"), log_file)
} else {
  log_message("ERROR: No valid predictions available for Phase 2", log_file)
  stop("Phase 1 failed to generate any valid predictions")
}

# Create player predictions sheet
player_predictions <- loocv_results %>%
  group_by(eventID) %>%
  arrange(eventID, desc(model_score), desc(rating)) %>%
  mutate(
    model_rank = row_number(),
    prediction_correct = ifelse(
      (model_rank <= 20 & actual_top20 == 1) | (model_rank > 20 & actual_top20 == 0), 
      1, 0
    )
  ) %>%
  ungroup() %>%
  select(eventID, playerID, model_score, model_rank, actual_top20, 
         actual_position, prediction_correct, top20_odds, top20_winnings, 
         rating, valid_odds) %>%
  arrange(eventID, desc(model_score))

# Save Phase 1 results
timestamp <- format(Sys.time(), "%m%d_%H%M")
phase1_file <- paste0("C:/Projects/Golf/temp_status/Phase1_Predictions_", timestamp, ".rds")
saveRDS(loocv_results, phase1_file)
log_message(paste("Phase 1 complete. Predictions saved to:", phase1_file), log_file)

# ===== PHASE 2: SEQUENTIAL SIMULATION =====
log_message("Starting Phase 2: Sequential Simulation", log_file)
cat("Starting Phase 2: Sequential Simulation...\n")

# Sort predictions by eventID (chronological proxy)
sorted_predictions <- loocv_results %>%
  filter(nrow(.) > 0) %>%  # Remove empty results
  arrange(eventID) %>%
  group_by(eventID) %>%
  nest(players = -eventID) %>%
  arrange(eventID)

log_message(paste("Sequential simulation on", nrow(sorted_predictions), "events"), log_file)

# Generate strategies
strategies <- generate_betting_strategies(log_file = log_file)

# Initialize bankroll tracking
strategy_bankrolls <- list()
strategy_histories <- list()

for(strategy_name in names(strategies)) {
  strategy_bankrolls[[strategy_name]] <- STARTING_BANKROLL
  strategy_histories[[strategy_name]] <- data.frame()
}

# Sequential event processing
for(event_idx in 1:nrow(sorted_predictions)) {
  event_data <- sorted_predictions$players[[event_idx]]
  current_eventID <- sorted_predictions$eventID[event_idx]
  
  if (event_idx %% 10 == 0) {
    log_message(paste("Sequential simulation: processed", event_idx, "of", nrow(sorted_predictions), "events"), log_file, console = FALSE)
  }
  
  # Process each strategy for this event
  for(strategy_name in names(strategies)) {
    strategy <- strategies[[strategy_name]]
    current_bankroll <- strategy_bankrolls[[strategy_name]]
    
    # Skip if bankroll depleted
    if(current_bankroll <= 0) next
    
    # Calculate bets for this strategy
    bet_results <- calculate_strategy_bets(event_data, strategy, FIXED_STAKE_AMOUNT)
    
    # Check if sufficient bankroll
    if(current_bankroll < bet_results$total_stake) {
      # Insufficient funds - strategy becomes inactive
      strategy_bankrolls[[strategy_name]] <- 0
      bet_results <- list(num_bets = 0, total_stake = 0, total_winnings = 0, profit_loss = 0)
    } else {
      # Update bankroll
      new_bankroll <- current_bankroll + bet_results$profit_loss
      strategy_bankrolls[[strategy_name]] <- max(0, new_bankroll)
    }

    # Record event details
    event_record <- data.frame(
      eventID = current_eventID,
      event_sequence = event_idx,
      strategy = strategy_name,
      bankroll_start = current_bankroll,
      num_bets = bet_results$num_bets,
      total_stake = bet_results$total_stake,
      total_winnings = bet_results$total_winnings,
      profit_loss = bet_results$profit_loss,
      bankroll_end = strategy_bankrolls[[strategy_name]],
      still_active = strategy_bankrolls[[strategy_name]] > 0,
      stringsAsFactors = FALSE
    )
    
    strategy_histories[[strategy_name]] <- rbind(strategy_histories[[strategy_name]], event_record)
  }
}

# ===== COMPILE RESULTS =====
log_message("Compiling final results...", log_file)

# Create final summary
Summary <- data.frame()
all_event_details <- data.frame()

for(strategy_name in names(strategies)) {
  history <- strategy_histories[[strategy_name]]
  
  if(nrow(history) > 0) {
    # Add to event details
    all_event_details <- rbind(all_event_details, history)
    
    # Calculate final metrics
    final_bankroll <- tail(history$bankroll_end, 1)
    total_profit <- final_bankroll - STARTING_BANKROLL
    total_roi <- (total_profit / STARTING_BANKROLL) * 100
    total_bets <- sum(history$num_bets)
    total_stake <- sum(history$total_stake)
    total_winnings <- sum(history$total_winnings)
    success_rate <- ifelse(total_bets > 0, sum(history$profit_loss > 0) / nrow(history) * 100, 0)
    
    summary_record <- data.frame(
      strategy = strategy_name,
      starting_bankroll = STARTING_BANKROLL,
      final_bankroll = round(final_bankroll, 2),
      total_profit = round(total_profit, 2),
      total_roi = round(total_roi, 2),
      total_bets = total_bets,
      total_stake = round(total_stake, 2),
      total_winnings = round(total_winnings, 2),
      events_participated = nrow(history),
      success_rate = round(success_rate, 2),
      still_active = final_bankroll > 0,
      max_bankroll = round(max(history$bankroll_end), 2),
      min_bankroll = round(min(history$bankroll_end), 2),
      stringsAsFactors = FALSE
    )
    
    Summary <- rbind(Summary, summary_record)
  }
}


# Save results
results_timestamp <- format(Sys.time(), "%m%d_%H%M")
saveRDS(Summary, paste0("C:/Projects/Golf/temp_status/Summary_", results_timestamp, ".rds"))
saveRDS(all_event_details, paste0("C:/Projects/Golf/temp_status/Event_Details_", results_timestamp, ".rds"))
saveRDS(strategy_histories, paste0("C:/Projects/Golf/temp_status/Bankroll_Histories_", results_timestamp, ".rds"))

log_message("Results saved to RDS files", log_file)

# Create Excel report
log_message("Creating Excel report...", log_file)
wb <- createWorkbook()

addWorksheet(wb, "Summary")
addWorksheet(wb, "Event_By_Event")
addWorksheet(wb, "Player_Predictions")

writeData(wb, "Summary", Summary)
writeData(wb, "Event_By_Event", all_event_details)
writeData(wb, "Player_Predictions", player_predictions)


# Add conditional formatting for ROI
negStyle <- createStyle(fontColour = "#9C0006", bgFill = "#FFC7CE")
posStyle <- createStyle(fontColour = "#006100", bgFill = "#C6EFCE")

roi_col <- which(names(Summary) == "total_roi")
if (length(roi_col) > 0) {
  conditionalFormatting(wb, "Summary", 
                        cols = roi_col, 
                        rows = 2:(nrow(Summary) + 1),
                        rule = "<0", style = negStyle)
  conditionalFormatting(wb, "Summary", 
                        cols = roi_col, 
                        rows = 2:(nrow(Summary) + 1),
                        rule = ">0", style = posStyle)
}

excel_file <- paste0("C:/Projects/Golf/Betting Simulations/LOOCV_Sequential_Analysis_", results_timestamp, ".xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)

log_message(paste("Excel report saved to:", excel_file), log_file)

# ===== DISPLAY RESULTS =====
# Log final summary statistics
log_message("=== FINAL ANALYSIS SUMMARY ===", log_file)
if (nrow(Summary) > 0) {
  log_message(paste("Total strategies tested:", nrow(Summary)), log_file)
  log_message(paste("Strategies still active:", sum(Summary$still_active)), log_file)
  log_message(paste("Best performing strategy:", Summary$strategy[which.max(Summary$total_roi)], 
                    "with", round(max(Summary$total_roi), 2), "% ROI"), log_file)
  log_message(paste("Worst performing strategy:", Summary$strategy[which.min(Summary$total_roi)], 
                    "with", round(min(Summary$total_roi), 2), "% ROI"), log_file)
  log_message(paste("Average ROI across all strategies:", round(mean(Summary$total_roi), 2), "%"), log_file)
  log_message(paste("Total events processed in simulation:", max(Summary$events_participated)), log_file)
} else {
  log_message("No strategies completed the analysis", log_file)
}

log_message("LOOCV Sequential Simulation completed successfully", log_file)
cat(sprintf("\n✓ Analysis complete! Results saved to:\n%s\n\n", excel_file))
