#### Enhanced Train/Test Split with Sequential Betting Simulation ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
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

# Train/test split parameters
TRAIN_TEST_SPLIT <- 0.5  # 50% for training, 50% for testing
RANDOM_SEED <- 42

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

# Feature engineering (run once at start)
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

# Modified model training for single train/test split
train_ensemble_model <- function(train_data, available_vars, log_file = NULL) {
  tryCatch({
    log_message("Training ensemble model on training data...", log_file)
    
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
      resp.name = paste0("PGA_TrainTest_", Sys.getpid()),
      dir.name = paste0(getwd(), "/Models")
    )
    
    # Create dummy coordinates
    dummy_coords <- data.frame(
      x = 1:length(myBiomodData@data.species),
      y = 1:length(myBiomodData@data.species)
    )
    myBiomodData@coord <- dummy_coords
    
    # Train individual models
    log_message("Training individual models...", log_file)
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
      do.progress = TRUE
    )
    
    # Evaluate models
    myBiomodModelEval <- get_evaluations(myBiomodModelOut)
    
    # Check if evaluation was successful
    if (is.null(myBiomodModelEval) || nrow(myBiomodModelEval) == 0) {
      log_message("Model evaluation failed", log_file)
      return(NULL)
    }
    
    allRun.model.names <- myBiomodModelEval[
      which(myBiomodModelEval$run == "allRun" & myBiomodModelEval$metric.eval == "TSS"),
      "full.name"
    ]
    
    if (length(allRun.model.names) == 0) {
      log_message("No valid models found", log_file)
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
    
    log_message("Creating ensemble model...", log_file)
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
    
    log_message("Ensemble model training completed successfully", log_file)
    return(myBiomodEM)
    
  }, error = function(e) {
    error_msg <- paste("Error in model training:", e$message)
    log_message(error_msg, log_file)
    return(NULL)
  })
}

# Generate predictions for test data
generate_predictions <- function(ensemble_model, test_data, available_vars, log_file = NULL) {
  tryCatch({
    log_message("Generating predictions on test data...", log_file)
    
    test_expl <- test_data[, available_vars]
    test_resp.xy <- test_data[, c("playerID", "eventID")]
    
    myBiomodProj <- BIOMOD_EnsembleForecasting(
      bm.em = ensemble_model,
      bm.proj = NULL,
      proj.name = "test_predictions",
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
      test_data[, c("eventID", "playerID", "Top40_Odds", "top40_winnings", "rating", "posn", "top_40")],
      by = c("eventID", "playerID"), all.x = TRUE
    )
    
    PGA_Prediction$Valid_Odds <- PGA_Prediction$Top40_Odds > 1
    
    log_message(paste("Generated", nrow(PGA_Prediction), "predictions"), log_file)
    return(PGA_Prediction)
    
  }, error = function(e) {
    error_msg <- paste("Error in prediction generation:", e$message)
    log_message(error_msg, log_file)
    return(NULL)
  })
}

# Calculate betting results for strategy
calculate_strategy_bets <- function(event_predictions, strategy, fixed_stake_amount) {
  # Rank players by model score
  event_predictions <- event_predictions %>%
    arrange(desc(Model_Score), desc(rating)) %>%
    mutate(rank_order = row_number())
  
  # Apply strategy range
  bet_players <- event_predictions %>%
    filter(rank_order >= strategy$start, 
           rank_order <= strategy$end,
           Valid_Odds == TRUE)
  
  if(nrow(bet_players) == 0) {
    return(list(num_bets = 0, total_stake = 0, total_winnings = 0, profit_loss = 0))
  }
  
  # Fixed stakes
  total_stake <- nrow(bet_players) * fixed_stake_amount
  total_winnings <- sum(ifelse(bet_players$top_40 == 1, 
                               bet_players$top40_winnings, 0))
  profit_loss <- total_winnings - total_stake
  
  return(list(
    num_bets = nrow(bet_players),
    total_stake = total_stake,
    total_winnings = total_winnings,
    profit_loss = profit_loss
  ))
}

# Split events into train/test
split_events_train_test <- function(unique_events, train_percentage, random_seed, log_file = NULL) {
  set.seed(random_seed)
  
  # Randomly sample events for training
  n_train_events <- floor(length(unique_events) * train_percentage)
  train_events <- sample(unique_events, n_train_events, replace = FALSE)
  test_events <- unique_events[!unique_events %in% train_events]
  
  log_message(paste("Train/Test split created:"), log_file)
  log_message(paste("  Training events:", length(train_events)), log_file)
  log_message(paste("  Test events:", length(test_events)), log_file)
  log_message(paste("  Train percentage:", round(length(train_events)/length(unique_events)*100, 1), "%"), log_file)
  
  return(list(train_events = train_events, test_events = test_events))
}

# ===== MAIN EXECUTION =====
# Create log file
log_file <- paste0("C:/Projects/Golf/temp_status/train_test_simulation_", format(Sys.time(), "%m%d_%H%M"), ".log")
log_message("Starting Train/Test Split with Sequential Betting Simulation", log_file)

# Load and prepare data
setwd("C:/Projects/Golf")
df <- read_excel("./Data/PGA_revised_odds.xlsx")
names(df) <- gsub("^_", "X_", names(df))
df$rating <- as.numeric(df$rating)
df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df[is.na(df$Top40_Profit), "Top40_Profit"] <- -10
df[is.na(df$Top20_Profit), "Top20_Profit"] <- -10
df$top40_winnings <- df$Top40_Profit + 10
df$Top20_Winnings <- df$Top20_Profit + 10
df <- df[complete.cases(df),]
df$eventID <- as.numeric(df$eventID)

eventQuality <- read.csv("./Data/FieldQuality.csv")
df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID"))

log_message(paste("Loaded data:", nrow(df), "rows,", length(unique(df$eventID)), "unique events"), log_file)

# Create derived features
df <- create_derived_features(df, log_file)

# Define model variables
model_vars <- c("log_rating", "current_rank", "compat_rank", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "sgapp", "Starts_Not10", "Top40_Odds")

available_vars <- model_vars[model_vars %in% names(df)]
log_message(paste("Available model variables:", length(available_vars), "of", length(model_vars)), log_file)

# Create model dataset
model_data <- df %>%
  select(playerID, eventID, posn, top_40, top40_winnings, rating, Top40_Odds, all_of(available_vars))

# Get unique events and split into train/test
unique_events <- unique(model_data$eventID)
event_split <- split_events_train_test(unique_events, TRAIN_TEST_SPLIT, RANDOM_SEED, log_file)

# ===== PHASE 1: MODEL TRAINING =====
log_message("Starting Phase 1: Model Training", log_file)
cat("Starting Phase 1: Model Training...\n")

# Create training and test datasets
train_data <- model_data %>% filter(eventID %in% event_split$train_events)
test_data <- model_data %>% filter(eventID %in% event_split$test_events)

log_message(paste("Training data:", nrow(train_data), "rows from", length(event_split$train_events), "events"), log_file)
log_message(paste("Test data:", nrow(test_data), "rows from", length(event_split$test_events), "events"), log_file)

# Train ensemble model
start_time_training <- Sys.time()
ensemble_model <- train_ensemble_model(train_data, available_vars, log_file)
training_time <- as.numeric(difftime(Sys.time(), start_time_training, units = "mins"))

if (is.null(ensemble_model)) {
  log_message("ERROR: Model training failed", log_file)
  stop("Model training failed")
}

log_message(paste("Model training completed in", round(training_time, 2), "minutes"), log_file)

# ===== PHASE 2: PREDICTION GENERATION =====
log_message("Starting Phase 2: Prediction Generation", log_file)
cat("Starting Phase 2: Prediction Generation...\n")

# Generate predictions on test data
start_time_prediction <- Sys.time()
test_predictions <- generate_predictions(ensemble_model, test_data, available_vars, log_file)
prediction_time <- as.numeric(difftime(Sys.time(), start_time_prediction, units = "mins"))

if (is.null(test_predictions) || nrow(test_predictions) == 0) {
  log_message("ERROR: Prediction generation failed", log_file)
  stop("Prediction generation failed")
}

log_message(paste("Prediction generation completed in", round(prediction_time, 2), "minutes"), log_file)

# Analyze prediction quality
score_summary <- summary(test_predictions$Model_Score)
log_message(paste("Model score distribution - Min:", round(score_summary[1], 3), 
                  "Q1:", round(score_summary[2], 3),
                  "Median:", round(score_summary[3], 3),
                  "Q3:", round(score_summary[5], 3),
                  "Max:", round(score_summary[6], 3)), log_file)

valid_odds_pct <- round(sum(test_predictions$Valid_Odds) / nrow(test_predictions) * 100, 1)
top40_rate <- round(sum(test_predictions$top_40) / nrow(test_predictions) * 100, 1)

log_message(paste("Valid odds percentage:", valid_odds_pct, "%"), log_file)
log_message(paste("Actual top 40 rate:", top40_rate, "%"), log_file)

# Create full testing predictions dataframe with specified columns
full_test_predictions <- test_predictions %>%
  select(eventID, playerID, Model_Score, rating, posn, top40_winnings) %>%
  rename(
    Event_ID = eventID,
    Player_ID = playerID,
    Model_Score = Model_Score,
    Rating = rating,
    Position = posn,
    Top40_Winnings = top40_winnings
  ) %>%
  arrange(Event_ID, desc(Model_Score))

log_message(paste("Full testing predictions dataframe created with", nrow(full_test_predictions), "rows"), log_file)

# Save predictions
timestamp <- format(Sys.time(), "%m%d_%H%M")
predictions_file <- paste0("C:/Projects/Golf/temp_status/Test_Predictions_", timestamp, ".rds")
saveRDS(test_predictions, predictions_file)

# Save full testing predictions dataframe
full_predictions_file <- paste0("C:/Projects/Golf/temp_status/Full_Test_Predictions_", timestamp, ".rds")
saveRDS(full_test_predictions, full_predictions_file)

log_message(paste("Predictions saved to:", predictions_file), log_file)
log_message(paste("Full testing predictions saved to:", full_predictions_file), log_file)

# ===== PHASE 3: SEQUENTIAL SIMULATION =====
log_message("Starting Phase 3: Sequential Simulation", log_file)
cat("Starting Phase 3: Sequential Simulation...\n")

# Sort predictions by eventID (chronological proxy)
sorted_predictions <- test_predictions %>%
  arrange(eventID) %>%
  group_by(eventID) %>%
  nest(.key = "players") %>%
  arrange(eventID)

log_message(paste("Sequential simulation on", nrow(sorted_predictions), "test events"), log_file)

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
final_summary <- data.frame()
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
    
    final_summary <- rbind(final_summary, summary_record)
  }
}

# Save results
results_timestamp <- format(Sys.time(), "%m%d_%H%M")
saveRDS(final_summary, paste0("C:/Projects/Golf/temp_status/Final_Summary_TrainTest_", results_timestamp, ".rds"))
saveRDS(all_event_details, paste0("C:/Projects/Golf/temp_status/Event_Details_TrainTest_", results_timestamp, ".rds"))
saveRDS(strategy_histories, paste0("C:/Projects/Golf/temp_status/Bankroll_Histories_TrainTest_", results_timestamp, ".rds"))
saveRDS(full_test_predictions, paste0("C:/Projects/Golf/temp_status/Full_Test_Predictions_TrainTest_", results_timestamp, ".rds"))

# Save train/test split info
split_info <- list(
  train_events = event_split$train_events,
  test_events = event_split$test_events,
  train_percentage = TRAIN_TEST_SPLIT,
  random_seed = RANDOM_SEED,
  training_time_mins = training_time,
  prediction_time_mins = prediction_time
)
saveRDS(split_info, paste0("C:/Projects/Golf/temp_status/TrainTest_Split_Info_", results_timestamp, ".rds"))

log_message("Results saved to RDS files", log_file)

# Create Excel report
log_message("Creating Excel report...", log_file)
wb <- createWorkbook()

addWorksheet(wb, "Final_Summary")
addWorksheet(wb, "Event_Details")
addWorksheet(wb, "Top_Strategies")
addWorksheet(wb, "Split_Info")
addWorksheet(wb, "Full_Test_Predictions")

writeData(wb, "Final_Summary", final_summary)
writeData(wb, "Event_Details", all_event_details)
writeData(wb, "Full_Test_Predictions", full_test_predictions)

# Top 10 strategies
top_strategies <- final_summary %>%
  arrange(desc(total_roi)) %>%
  head(10)

writeData(wb, "Top_Strategies", top_strategies)

# Split information
split_summary <- data.frame(
  Metric = c("Training Events", "Test Events", "Total Events", "Train Percentage", 
             "Random Seed", "Training Time (mins)", "Prediction Time (mins)"),
  Value = c(length(event_split$train_events), length(event_split$test_events), 
            length(unique_events), paste0(TRAIN_TEST_SPLIT * 100, "%"),
            RANDOM_SEED, round(training_time, 2), round(prediction_time, 2))
)
writeData(wb, "Split_Info", split_summary)

# Add conditional formatting for ROI
negStyle <- createStyle(fontColour = "#9C0006", bgFill = "#FFC7CE")
posStyle <- createStyle(fontColour = "#006100", bgFill = "#C6EFCE")

roi_col <- which(names(final_summary) == "total_roi")
if (length(roi_col) > 0) {
  conditionalFormatting(wb, "Final_Summary", 
                        cols = roi_col, 
                        rows = 2:(nrow(final_summary) + 1),
                        rule = "<0", style = negStyle)
  conditionalFormatting(wb, "Final_Summary", 
                        cols = roi_col, 
                        rows = 2:(nrow(final_summary) + 1),
                        rule = ">0", style = posStyle)
}

excel_file <- paste0("C:/Projects/Golf/Betting Simulations/TrainTest_Sequential_Analysis_", results_timestamp, ".xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)

log_message(paste("Excel report saved to:", excel_file), log_file)

# ===== DISPLAY RESULTS =====
cat("\nTRAIN/TEST SPLIT SUMMARY:\n")
cat(sprintf("Training: %d events | Test: %d events | Split: %.0f%%/%.0f%%\n", 
            length(event_split$train_events), length(event_split$test_events),
            TRAIN_TEST_SPLIT * 100, (1 - TRAIN_TEST_SPLIT) * 100))
cat(sprintf("Training time: %.2f mins | Prediction time: %.2f mins\n\n", 
            training_time, prediction_time))

cat("TOP 10 PERFORMING STRATEGIES:\n")
if (nrow(top_strategies) > 0) {
  for (i in 1:min(10, nrow(top_strategies))) {
    cat(sprintf("%2d. %s: %.1f%% ROI (£%.2f final) - %d bets, %.1f%% success rate %s\n",
                i,
                top_strategies$strategy[i],
                top_strategies$total_roi[i],
                top_strategies$final_bankroll[i],
                top_strategies$total_bets[i],
                top_strategies$success_rate[i],
                ifelse(top_strategies$still_active[i], "✓", "✗")))
  }
} else {
  cat("No strategies generated results.\n")
}

# Log final summary statistics
log_message("=== FINAL ANALYSIS SUMMARY ===", log_file)
if (nrow(final_summary) > 0) {
  log_message(paste("Total strategies tested:", nrow(final_summary)), log_file)
  log_message(paste("Strategies still active:", sum(final_summary$still_active)), log_file)
  log_message(paste("Best performing strategy:", final_summary$strategy[which.max(final_summary$total_roi)], 
                    "with", round(max(final_summary$total_roi), 2), "% ROI"), log_file)
  log_message(paste("Worst performing strategy:", final_summary$strategy[which.min(final_summary$total_roi)], 
                    "with", round(min(final_summary$total_roi), 2), "% ROI"), log_file)
  log_message(paste("Average ROI across all strategies:", round(mean(final_summary$total_roi), 2), "%"), log_file)
  log_message(paste("Total test events processed:", max(final_summary$events_participated)), log_file)
} else {
  log_message("No strategies completed the analysis", log_file)
}

log_message("Train/Test Sequential Simulation completed successfully", log_file)
cat(sprintf("\n✓ Analysis complete! Results saved to:\n%s\n\n", excel_file))