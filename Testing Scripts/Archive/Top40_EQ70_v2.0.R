#### Enhanced 80/20 Split with Sequential Betting Simulation ####
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
MIN_PLAYERS_PER_EVENT <- 80

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'GLM', 'RF', 'FDA', 'CTA', 'ANN', 'GBM', 'MAXNET', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.75

# Train/Test split configuration
TRAIN_PERCENTAGE <- 0.80  # 80% for training, 20% for testing
SEED <- 123  # For reproducible splits

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
  
  # Create top 40 target variable
  df$top_40 <- ifelse(df$posn <= 40, 1, 0)
  
  # Event-level features processing
  events <- split(df, df$eventID)
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

# Model training function for single train/test split
train_ensemble_model_split <- function(train_data, available_vars, log_file = NULL) {
  tryCatch({
    resp <- train_data[, "top_40"]
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
      resp.name = paste0("PGA_8020_", Sys.getpid(), "_", sample(10000, 1)),
      dir.name = paste0(getwd(), "/Models")
    )
    
    # Create dummy coordinates
    dummy_coords <- data.frame(
      x = 1:length(myBiomodData@data.species),
      y = 1:length(myBiomodData@data.species)
    )
    myBiomodData@coord <- dummy_coords
    
    # Train individual models
    myBiomodModelOut <- BIOMOD_Modeling(
      bm.format = myBiomodData,
      models = MODELS_TO_PROCESS,
      OPT.strategy = "bigboss",
      CV.nb.rep = CV_REPETITIONS,
      CV.perc = CV_PERCENTAGE,
      weights = NULL,
      var.import = 0,
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
    
    # Create ensemble model
    myBiomodEM <- BIOMOD_EnsembleModeling(
      bm.mod = myBiomodModelOut,
      models.chosen = names(df.metrics),
      em.algo = c('EMwmean'),
      metric.select = c("user.defined"),
      metric.select.thresh = c(0.3),
      metric.select.table = df.metrics,
      metric.eval = c('TSS'),
      var.import = 0,
      EMwmean.decay = 'proportional'
    )
    
    return(myBiomodEM)
    
  }, error = function(e) {
    error_msg <- paste("Error in model training:", e$message)
    if (!is.null(log_file)) {
      tryCatch({
        cat(paste0(Sys.time(), " - ", error_msg, "\n"), file = log_file, append = TRUE)
      }, error = function(e2) {})
    }
    return(NULL)
  })
}

# Generate predictions for test events (parallel processing)
generate_predictions_parallel <- function(ensemble_model, test_events, model_data, available_vars, log_file = NULL) {
  
  # Set up parallel processing
  n_cores <- 12
  cl <- makeCluster(n_cores, type = "SOCK")
  registerDoSNOW(cl)
  
  log_message(paste("Setting up parallel processing with", n_cores, "cores for prediction"), log_file)
  
  # Progress tracking for predictions
  completed_events <- 0
  start_time <- Sys.time()
  
  progress <- function(n) {
    completed_events <<- completed_events + 1
    elapsed_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    if (completed_events > 1) {
      avg_time_per_event <- elapsed_time / completed_events
      remaining_events <- length(test_events) - completed_events
      eta_seconds <- avg_time_per_event * remaining_events
      
      cat(paste0("Prediction Progress: ", completed_events, "/", length(test_events), 
                 " events | ETA: ", round(eta_seconds, 1), " seconds\n"))
    }
    flush.console()
  }
  
  opts <- list(progress = progress)
  
  # Export variables to workers
  clusterExport(cl, c("ensemble_model", "model_data", "available_vars", "log_file"))
  
  clusterEvalQ(cl, {
    library(biomod2)
    library(dplyr)
  })
  
  log_message(paste("Starting parallel prediction generation for", length(test_events), "events"), log_file)
  
  # Parallel prediction loop
  prediction_results <- foreach(event_id = test_events, 
                                .packages = c('biomod2', 'dplyr'),
                                .options.snow = opts,
                                .errorhandling = 'pass',
                                .combine = 'rbind') %dopar% {
                                  
                                  tryCatch({
                                    # Get test data for this event
                                    test_data <- model_data[model_data$eventID == event_id, ]
                                    
                                    if (nrow(test_data) < 5) {  # Skip events with too few players
                                      return(NULL)
                                    }
                                    
                                    test_expl <- test_data[, available_vars]
                                    test_resp.xy <- test_data[, c("playerID", "eventID")]
                                    
                                    # Generate predictions
                                    myBiomodProj <- BIOMOD_EnsembleForecasting(
                                      bm.em = ensemble_model,
                                      bm.proj = NULL,
                                      proj.name = paste0("event_", event_id),
                                      new.env = test_expl,
                                      new.env.xy = test_resp.xy,
                                      models.chosen = "all",
                                      metric.binary = "TSS",
                                      metric.filter = "TSS",
                                      na.rm = TRUE
                                    )
                                    
                                    # Format predictions
                                    PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
                                    colnames(PGA_Prediction)[3] <- "Model_Score"
                                    
                                    PGA_Prediction <- merge(
                                      PGA_Prediction,
                                      test_data[, c("eventID", "playerID", "Top40_Odds", "top40_winnings", "rating", "posn", "top_40")],
                                      by = c("eventID", "playerID"), all.x = TRUE
                                    )
                                    
                                    PGA_Prediction$Valid_Odds <- PGA_Prediction$Top40_Odds > 1
                                    
                                    # Return structured data
                                    result_data <- data.frame(
                                      eventID = event_id,
                                      playerID = PGA_Prediction$playerID,
                                      model_score = PGA_Prediction$Model_Score,
                                      actual_top40 = PGA_Prediction$top_40,
                                      actual_position = PGA_Prediction$posn,
                                      top40_odds = PGA_Prediction$Top40_Odds,
                                      top40_winnings = PGA_Prediction$top40_winnings,
                                      rating = PGA_Prediction$rating,
                                      valid_odds = PGA_Prediction$Valid_Odds,
                                      stringsAsFactors = FALSE
                                    )
                                    
                                    return(result_data)
                                    
                                  }, error = function(e) {
                                    return(NULL)
                                  })
                                }
  
  stopCluster(cl)
  
  log_message("Parallel prediction generation completed", log_file)
  return(prediction_results)
}

# Calculate betting results for strategy
calculate_strategy_bets <- function(event_predictions, strategy, fixed_stake_amount) {
  # Rank players by model score
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
  total_winnings <- sum(ifelse(bet_players$actual_top40 == 1, 
                               bet_players$top40_winnings, 0))
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
log_file <- paste0("C:/Projects/Golf/temp_status/8020_sequential_simulation_", format(Sys.time(), "%m%d_%H%M"), ".log")
log_message("Starting 80/20 Split with Sequential Betting Simulation", log_file)

# Load and prepare data
setwd("C:/Projects/Golf")
df <- read_excel("./Data/PGA_revised_odds.xlsx")
names(df) <- gsub("^_", "X_", names(df))
df$rating <- as.numeric(df$rating)
df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df$top40_winnings <- df$Top40_Profit + 10
df$Top20_Winnings <- df$Top20_Profit + 10
df <- df[complete.cases(df),]
df$eventID <- as.numeric(df$eventID)

eventQuality <- read.csv("./Data/FieldQuality.csv")
df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID"))

# Create derived features
df <- create_derived_features(df, log_file)

# Define model variables
model_vars <- c("field","log_rating", "current_rank", "compat_rank", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "sgapp", "Starts_Not10", "Top40_Odds")

available_vars <- model_vars[model_vars %in% names(df)]
log_message(paste("Available model variables:", length(available_vars), "of", length(model_vars)), log_file)

# Create model dataset
model_data <- df %>%
  select(playerID, eventID, posn, top_40, top40_winnings, rating, all_of(available_vars)) %>%
  filter(!is.na(eventID))

# Get unique events and create 80/20 split
set.seed(SEED)
unique_events <- unique(model_data$eventID)
log_message(paste("Total unique events:", length(unique_events)), log_file)

# Create chronological split (first 80% for training, last 20% for testing)
sorted_events <- sort(unique_events)
n_train_events <- floor(length(sorted_events) * TRAIN_PERCENTAGE)
train_events <- sorted_events[1:n_train_events]
test_events <- sorted_events[(n_train_events + 1):length(sorted_events)]

log_message(paste("Training events:", length(train_events), "(", round(length(train_events)/length(sorted_events)*100, 1), "%)"), log_file)
log_message(paste("Testing events:", length(test_events), "(", round(length(test_events)/length(sorted_events)*100, 1), "%)"), log_file)

# ===== PHASE 1: MODEL TRAINING =====
log_message("Starting Phase 1: Model Training", log_file)
cat("Starting Phase 1: Model Training...\n")

start_time_training <- Sys.time()

# Prepare training data
train_data <- model_data[model_data$eventID %in% train_events, ]
log_message(paste("Training data:", nrow(train_data), "observations from", length(train_events), "events"), log_file)

# Train ensemble model
log_message("Training ensemble model on 80% of events...", log_file)
ensemble_model <- train_ensemble_model_split(train_data, available_vars, log_file)

if (is.null(ensemble_model)) {
  log_message("ERROR: Model training failed", log_file)
  stop("Model training failed")
}

training_time <- as.numeric(difftime(Sys.time(), start_time_training, units = "mins"))
log_message(paste("Model training completed in", round(training_time, 2), "minutes"), log_file)

# ===== PHASE 2: PREDICTION GENERATION =====
log_message("Starting Phase 2: Prediction Generation", log_file)
cat("Starting Phase 2: Prediction Generation...\n")

start_time_prediction <- Sys.time()

# Filter test events by minimum players
test_event_sizes <- model_data %>%
  filter(eventID %in% test_events) %>%
  group_by(eventID) %>%
  summarise(n_players = n(), .groups = "drop")

valid_test_events <- test_event_sizes %>%
  filter(n_players >= MIN_PLAYERS_PER_EVENT) %>%
  pull(eventID)

log_message(paste("Valid test events (>= ", MIN_PLAYERS_PER_EVENT, " players):", length(valid_test_events), "of", length(test_events)), log_file)

if (length(valid_test_events) == 0) {
  log_message("ERROR: No valid test events available", log_file)
  stop("No valid test events available")
}

# Generate predictions for test events
predictions <- generate_predictions_parallel(ensemble_model, valid_test_events, model_data, available_vars, log_file)

if (is.null(predictions) || nrow(predictions) == 0) {
  log_message("ERROR: Prediction generation failed", log_file)
  stop("Prediction generation failed")
}

prediction_time <- as.numeric(difftime(Sys.time(), start_time_prediction, units = "mins"))
log_message(paste("Prediction generation completed in", round(prediction_time, 2), "minutes"), log_file)
log_message(paste("Generated", nrow(predictions), "predictions for", length(unique(predictions$eventID)), "events"), log_file)

# Create player predictions sheet
player_predictions <- predictions %>%
  group_by(eventID) %>%
  arrange(eventID, desc(model_score), desc(rating)) %>%
  mutate(
    model_rank = row_number(),
    prediction_correct = ifelse(
      (model_rank <= 40 & actual_top40 == 1) | (model_rank > 40 & actual_top40 == 0), 
      1, 0
    )
  ) %>%
  ungroup() %>%
  select(eventID, playerID, model_score, model_rank, actual_top40, 
         actual_position, prediction_correct, top40_odds, top40_winnings, 
         rating, valid_odds) %>%
  arrange(eventID, desc(model_score))

# Save Phase 2 results
timestamp <- format(Sys.time(), "%m%d_%H%M")
phase2_file <- paste0("C:/Projects/Golf/temp_status/Phase2_Predictions_", timestamp, ".rds")
saveRDS(predictions, phase2_file)
log_message(paste("Phase 2 complete. Predictions saved to:", phase2_file), log_file)

# ===== PHASE 3: SEQUENTIAL SIMULATION =====
log_message("Starting Phase 3: Sequential Simulation", log_file)
cat("Starting Phase 3: Sequential Simulation...\n")

# Sort predictions by eventID (chronological)
sorted_predictions <- predictions %>%
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
  
  if (event_idx %% 5 == 0) {
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
saveRDS(Summary, paste0("C:/Projects/Golf/temp_status/Summary_8020_", results_timestamp, ".rds"))
saveRDS(all_event_details, paste0("C:/Projects/Golf/temp_status/Event_Details_8020_", results_timestamp, ".rds"))
saveRDS(strategy_histories, paste0("C:/Projects/Golf/temp_status/Bankroll_Histories_8020_", results_timestamp, ".rds"))

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

excel_file <- paste0("C:/Projects/Golf/Betting Simulations/8020_Sequential_Analysis_", results_timestamp, ".xlsx")
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
  
  # Additional performance metrics
  log_message(paste("Training/Test split:", TRAIN_PERCENTAGE*100, "/", (1-TRAIN_PERCENTAGE)*100), log_file)
  log_message(paste("Total processing time:", round(training_time + prediction_time, 2), "minutes"), log_file)
  
  # Model accuracy metrics
  if (nrow(player_predictions) > 0) {
    overall_accuracy <- round(sum(player_predictions$prediction_correct) / nrow(player_predictions) * 100, 2)
    top40_actual_rate <- round(sum(player_predictions$actual_top40) / nrow(player_predictions) * 100, 2)
    valid_odds_rate <- round(sum(player_predictions$valid_odds) / nrow(player_predictions) * 100, 2)
    
    log_message(paste("Model prediction accuracy:", overall_accuracy, "%"), log_file)
    log_message(paste("Actual top-40 rate:", top40_actual_rate, "%"), log_file)
    log_message(paste("Valid odds availability:", valid_odds_rate, "%"), log_file)
  }
} else {
  log_message("No strategies completed the analysis", log_file)
}

log_message("80/20 Split Sequential Simulation completed successfully", log_file)
cat(sprintf("\n✓ Analysis complete! Results saved to:\n%s\n\n", excel_file))

# ===== COMPARISON OUTPUT =====
cat("=== ANALYSIS SUMMARY ===\n")
cat(sprintf("Model Training: %s minutes\n", round(training_time, 2)))
cat(sprintf("Prediction Generation: %s minutes\n", round(prediction_time, 2)))
cat(sprintf("Total Processing Time: %s minutes\n", round(training_time + prediction_time, 2)))
cat(sprintf("Training Events: %s (%s%%)\n", length(train_events), round(length(train_events)/length(unique_events)*100, 1)))
cat(sprintf("Test Events: %s (%s%%)\n", length(valid_test_events), round(length(valid_test_events)/length(unique_events)*100, 1)))
cat(sprintf("Total Predictions Generated: %s\n", nrow(predictions)))
cat(sprintf("Strategies Tested: %s\n", nrow(Summary)))

if (nrow(Summary) > 0) {
  cat(sprintf("Best Strategy: %s (%.2f%% ROI)\n", Summary$strategy[which.max(Summary$total_roi)], max(Summary$total_roi)))
  cat(sprintf("Worst Strategy: %s (%.2f%% ROI)\n", Summary$strategy[which.min(Summary$total_roi)], min(Summary$total_roi)))
  cat(sprintf("Average ROI: %.2f%%\n", mean(Summary$total_roi)))
  cat(sprintf("Strategies Still Active: %s of %s\n", sum(Summary$still_active), nrow(Summary)))
}