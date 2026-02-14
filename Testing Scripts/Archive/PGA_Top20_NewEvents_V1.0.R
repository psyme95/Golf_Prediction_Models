#### Enhanced Sequential Betting Strategy Simulation - FIXED STAKES ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(openxlsx)
library(ggplot2)
library(gridExtra)
library(readxl)

'%not in%' <- function(x, table) is.na(match(x, table, nomatch = NA_integer_))

# ===== CONFIGURATION SECTION =====
# Set working directory and parameters
setwd("C:/Projects/Golf")
#set.seed(123)

# Simulation parameters - moved to top for easy configuration
STARTING_BANKROLL <- 1000
FIXED_STAKE <- 10  # Fixed £10 per bet (changed from percentage)
MIN_PLAYERS_PER_EVENT <- 1

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'GLM', 'RF', 'FDA', 'CTA', 'ANN', 'GBM', 'MAXNET', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.75

# ===== HELPER FUNCTIONS =====
# Enhanced logging function
log_message <- function(message, log_file = NULL, console = TRUE, timestamp = TRUE) {
  if (timestamp) {
    full_message <- paste0(Sys.time(), " - ", message, "\n")
  } else {
    full_message <- paste0(message, "\n")
  }
  
  if (console) cat(full_message)
  if (!is.null(log_file)) cat(full_message, file = log_file, append = TRUE)
}

# Improved data validation function
validate_data <- function(df, data_name, required_columns = NULL) {
  log_message(paste("Validating", data_name, "data..."))
  
  # Basic validation
  if (nrow(df) == 0) {
    stop(paste(data_name, "dataset is empty"))
  }
  
  # Check for required columns
  if (!is.null(required_columns)) {
    missing_cols <- required_columns[!required_columns %in% names(df)]
    if (length(missing_cols) > 0) {
      warning(paste("Missing columns in", data_name, ":", paste(missing_cols, collapse = ", ")))
    }
  }
  
  # Check for NA values in key columns
  na_summary <- df %>%
    summarise_all(~ sum(is.na(.))) %>%
    select_if(~ . > 0)
  
  if (ncol(na_summary) > 0) {
    log_message(paste("NA values found in", data_name, ":"))
    print(na_summary)
  }
  
  log_message(paste(data_name, "validation complete:", nrow(df), "rows,", ncol(df), "columns"))
  return(df)
}

# Enhanced feature engineering function
create_derived_features <- function(df, log_file = NULL) {
  log_message("Creating derived features...", log_file)
  
  # Create top 20 target variable
  df$top_20 <- ifelse(df$posn <= 20, 1, 0)
  
  # Performance trend calculation with error handling
  if (all(c("current", "X_6m", "X_1yr", "yr3_All") %in% names(df))) {
    df$performance_trend <- sapply(1:nrow(df), function(i) {
      tryCatch({
        time_points <- c(0, 1, 2, 3)
        scores <- c(df$current[i], df$X_6m[i], df$X_1yr[i], df$yr3_All[i])
        
        valid_indices <- !is.na(scores)
        if (sum(valid_indices) >= 3) {
          valid_times <- time_points[valid_indices]
          valid_scores <- scores[valid_indices]
          
          if (length(valid_scores) >= 2) {
            return(lm(valid_scores ~ valid_times)$coefficients[2])
          }
        }
        return(NA)
      }, error = function(e) NA)
    })
    log_message("Performance trend feature created successfully", log_file)
  } else {
    df$performance_trend <- NA
    missing_vars <- c("current", "X_6m", "X_1yr", "yr3_All")[!c("current", "X_6m", "X_1yr", "yr3_All") %in% names(df)]
    log_message(paste("Warning: Performance trend variables missing:", paste(missing_vars, collapse = ", ")), log_file)
  }
  
  # Course advantage calculation with improved error handling
  if (all(c("course", "yr3_All") %in% names(df))) {
    df$course <- as.numeric(df$course)
    df$yr3_All <- as.numeric(df$yr3_All)
    
    df$course_advantage <- ifelse(
      !is.na(df$course) & !is.na(df$yr3_All) & df$yr3_All != 0,
      df$course / df$yr3_All,
      NA
    )
    log_message("Course advantage feature created successfully", log_file)
  } else {
    df$course_advantage <- NA
    missing_vars <- c("course", "yr3_All")[!c("course", "yr3_All") %in% names(df)]
    log_message(paste("Warning: Course advantage variables missing:", paste(missing_vars, collapse = ", ")), log_file)
  }
  
  # Event-level features with progress tracking
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
    event_data$log_rating <- as.numeric(log(event_data$rating+50))
    
    df_with_diffs <- rbind(df_with_diffs, event_data)
  }
  
  log_message("Derived features creation completed", log_file)
  return(df_with_diffs)
}

# Enhanced strategy generation function with validation
generate_betting_strategies <- function(size_increments = c(10, 20, 30, 40), log_file = NULL) {
  log_message("Generating betting strategies...", log_file)
  
  ranges_df <- data.frame()
  
  for (size in size_increments) {
    max_start <- 81 - size  # Ensure we don't go beyond rank 100
    for (start_val in seq(1, max_start, by = 10)) {
      end_val <- start_val + size - 1
      
      # Validate strategy makes sense
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
  log_message(paste("Range sizes:", paste(size_increments, collapse = ", ")), log_file)
  
  # Log some example strategies for validation
  example_strategies <- head(names(strategies), 5)
  for (strategy_name in example_strategies) {
    strategy <- strategies[[strategy_name]]
    log_message(paste("Example:", strategy_name, "-", strategy$description), log_file, console = FALSE)
  }
  
  return(strategies)
}

# Enhanced model training function with better error handling
train_ensemble_model <- function(train_data, model_vars, log_file = NULL) {
  log_message("Training ensemble model...", log_file)
  
  tryCatch({
    resp <- train_data[, "top_20"]
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
    
    # Train individual models
    myBiomodModelOut <- BIOMOD_Modeling(
      bm.format = myBiomodData,
      models = MODELS_TO_PROCESS,
      OPT.strategy = "bigboss",
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
      metric.select.thresh = c(0.3),
      metric.select.table = df.metrics,
      metric.eval = c('TSS'),
      var.import = 1,
      EMwmean.decay = 'proportional'
    )
    
    log_message("Model training completed successfully", log_file)
    return(myBiomodEM)
    
  }, error = function(e) {
    log_message(paste("Error in model training:", e$message), log_file)
    stop(e)
  })
}

# ===== MAIN EXECUTION =====
# Create log file
log_file <- paste0("C:/Projects/Golf/temp_status/betting_simulation_fixed_stakes_", format(Sys.time(), "%m%d_%H%M"), ".log")
log_message("Starting Sequential Betting Simulation - FIXED £10 STAKES", log_file)

# Load and validate data (KEEP QUALITY DATA FOR INFORMATION PURPOSES)
eventQuality <- read.csv("./Data/FieldQuality.csv")

log_message("Loading training data...", log_file)
df_old <- read_excel("./Data/PGA_revised_odds.xlsx")
names(df_old) <- gsub("^_", "X_", names(df_old))
df_old$rating <- as.numeric(df_old$rating)
df_old[is.na(df_old$Top40_Odds), "Top40_Odds"] <- 0
df_old[is.na(df_old$Top20_Odds), "Top20_Odds"] <- 0
df_old <- df_old[complete.cases(df_old), ]
df_old$eventID <- as.numeric(df_old$eventID)
df_old <- df_old %>% 
  left_join(eventQuality, by = c("eventID" = "EventID"))

df_old <- validate_data(df_old, "training")
df_old <- create_derived_features(df_old, log_file)

# Split events into list elements and randomly select 20% for testing
event_list <- split(df_old, df_old$eventID)
event_list <- event_list[sapply(event_list, function(df) nrow(df) > 1)]

# Combine all tournament data into one dataframe
df_old <- data.frame()

for (eventID in names(event_list)) {
  temp_df <- event_list[[eventID]]
  temp_df$eventID <- eventID
  df_old <- rbind(df_old, temp_df)
}

log_message("Loading test data...", log_file)
df_new <- read.csv("./Data/PGA_update_1706.csv")
df_new$rating <- as.numeric(df_new$rating)
# df_new[is.na(df_new$Top40_Odds), "Top40_Odds"] <- 0
# df_new[is.na(df_new$Top20_Odds), "Top20_Odds"] <- 0
# df_new[is.na(df_new$Top40_Profit), "Top40_Profit"] <- 0
# df_new[is.na(df_new$Top20_Profit), "Top20_Profit"] <- 0
df_new <- select(df_new, 1:70, "Top20_Odds", "Top20_Profit")
df_new <- df_new[complete.cases(df_new), ]
df_new$eventID <- as.numeric(df_new$eventID)
df_new <- df_new %>%
  mutate(top_20 = ifelse(posn <= 20, 1, 0))
df_new <- validate_data(df_new, "test")
df_new <- create_derived_features(df_new, log_file)

# Define model variables
model_vars <- c("field", "current_rank", "compat_rank", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "sgapp", "Starts_Not10", "diff_from_max", "current_top20", "compat2",
                "AvPosn_Rank", "yr3_All_rank", "performance_trend")

# Check available variables
available_vars_old <- model_vars[model_vars %in% names(df_old)]
available_vars_new <- model_vars[model_vars %in% names(df_new)]
available_vars <- intersect(available_vars_old, available_vars_new)

log_message(paste("Available model variables:", length(available_vars), "of", length(model_vars)), log_file)

# Create datasets (KEEP QUALITY FOR INFORMATION, BUT DON'T USE IN SIMULATION)
train_data <- df_old %>%
  select(playerID, eventID, posn, top_20, all_of(available_vars))

test_data <- df_new %>%
  select(playerID, eventID, posn, top_20, Quality, Top20_Profit, rating, all_of(available_vars))

log_message(paste("Training data:", nrow(train_data), "observations from", 
                  length(unique(train_data$eventID)), "events"), log_file)
log_message(paste("Test data:", nrow(test_data), "observations from", 
                  length(unique(test_data$eventID)), "events"), log_file)

# Generate betting strategies
strategies <- generate_betting_strategies(log_file = log_file)

# Train ensemble model
myBiomodEM <- train_ensemble_model(train_data, available_vars, log_file)

# Initialize simulations
new_events <- unique(test_data$eventID)

# Create event information (including quality metrics for reference)
event_info <- test_data %>%
  select(eventID, Quality) %>%
  distinct() %>%
  left_join(
    test_data %>%
      group_by(eventID) %>%
      summarize(Number_of_Players = n(), .groups = "drop"),
    by = "eventID"
  )

# Initialize prediction tracking for detailed analysis
all_predictions <- data.frame()

log_message(paste("Processing", length(new_events), "events"), log_file)

# Initialize simulation tracking - ONLY FIXED STAKES, NO QUALITY FILTERING
simulation_results <- list()

for (strategy_name in names(strategies)) {
  sim_key <- paste0("NoQuality_", strategy_name, "_Fixed")
  simulation_results[[sim_key]] <- list(
    approach = "NoQuality",
    strategy = strategy_name,
    type = "Fixed",
    quality_filter = "All",
    bankroll = STARTING_BANKROLL,
    history = data.frame(),
    active = TRUE,
    total_bets = 0,
    successful_bets = 0,
    total_staked = 0,
    total_winnings = 0,
    max_bankroll = STARTING_BANKROLL,
    max_drawdown = 0
  )
}

log_message(paste("Total simulations to run:", length(simulation_results)), log_file)

# Run event-by-event simulation
log_message(paste("Running simulation on", length(new_events), "events in sequence"), log_file)

for (event_idx in 1:length(new_events)) {
  test_event <- new_events[event_idx]
  
  if (event_idx %% 5 == 0) {
    log_message(paste("Processing event", event_idx, "of", length(new_events), ":", test_event), log_file, console = FALSE)
  }
  
  # Get event data
  event_data <- test_data %>%
    filter(eventID == test_event) %>%
    mutate(Top20_Odds = filter(df_new, eventID == test_event)$Top20_Odds,
           Top20_Profit = filter(df_new, eventID == test_event)$Top20_Profit,
           rating = filter(df_new, eventID == test_event)$rating)
  
  event_info_row <- event_info[event_info$eventID == test_event, ]
  
  if (nrow(event_data) < MIN_PLAYERS_PER_EVENT) {
    log_message(paste("Skipping event", test_event, "- insufficient players (", nrow(event_data), ")"), log_file, console = FALSE)
    next
  }
  
  # Generate predictions for this event
  tryCatch({
    test_expl <- event_data[, available_vars]
    test_resp.xy <- event_data[, c("playerID", "eventID")]
    
    myBiomodProj <- BIOMOD_EnsembleForecasting(
      bm.em = myBiomodEM,
      bm.proj = NULL,
      proj.name = paste0("event_", test_event),
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
      event_data[, c("eventID", "playerID", "Top20_Odds", "Top20_Profit", "posn", "top_20", "rating")],
      by = c("eventID", "playerID"), all.x = TRUE
    )
    
    PGA_Prediction$Valid_Odds <- PGA_Prediction$Top20_Odds > 1
    
    # Create detailed prediction record for this event
    PGA_Prediction$Model_Rank <- rank(-PGA_Prediction$Model_Score, ties.method = "first")
    PGA_Prediction$Prediction_Correct <- ifelse(
      (PGA_Prediction$top_20 == 1 & PGA_Prediction$Model_Rank <= 20) | 
        (PGA_Prediction$top_20 == 0 & PGA_Prediction$Model_Rank > 20), 
      1, 0)    
    PGA_Prediction$Top20_Winnings <- ifelse(PGA_Prediction$top_20 == 1, PGA_Prediction$Top20_Profit, -10)
    
    # Add to master predictions dataframe
    prediction_record <- data.frame(
      EventID = PGA_Prediction$eventID,
      PlayerID = PGA_Prediction$playerID,
      Model_Score = round(PGA_Prediction$Model_Score, 4),
      Model_Rank = PGA_Prediction$Model_Rank,
      Actual_Top20 = PGA_Prediction$top_20,
      Actual_Position = PGA_Prediction$posn,
      Prediction_Correct = PGA_Prediction$Prediction_Correct,
      Top20_Odds = PGA_Prediction$Top20_Odds,
      Top20_Winnings = PGA_Prediction$Top20_Winnings + 10,
      Top20_Profit = PGA_Prediction$Top20_Winnings,
      Rating = PGA_Prediction$rating,
      Valid_Odds = PGA_Prediction$Valid_Odds
    )
    
    all_predictions <- rbind(all_predictions, prediction_record)
    
    # Process each strategy for this event
    for (strategy_name in names(strategies)) {
      strategy <- strategies[[strategy_name]]
      
      # Calculate betting thresholds
      sorted_scores <- sort(PGA_Prediction$Model_Score, decreasing = TRUE)
      total_players <- nrow(PGA_Prediction)
      
      # Ensure we don't exceed available players
      actual_start_rank <- min(strategy$start, total_players)
      actual_end_rank <- min(strategy$end, total_players)
      
      # Get threshold scores for the rank range
      start_threshold <- sorted_scores[actual_start_rank]  # Highest score in range
      end_threshold <- sorted_scores[actual_end_rank]      # Lowest score in range
      
      # Select players whose scores fall within the rank range
      bet_players <- PGA_Prediction$Model_Score >= end_threshold & 
        PGA_Prediction$Model_Score <= start_threshold & 
        PGA_Prediction$Valid_Odds
      
      # Additional validation: ensure we're not betting on too many players
      if (sum(bet_players, na.rm = TRUE) > (actual_end_rank - actual_start_rank + 1)) {
        PGA_Prediction$rank_order <- rank(-PGA_Prediction$Model_Score, ties.method = "first")
        bet_players <- PGA_Prediction$rank_order >= actual_start_rank & 
          PGA_Prediction$rank_order <= actual_end_rank & 
          PGA_Prediction$Valid_Odds
      }
      
      bet_data <- PGA_Prediction[bet_players, ]
      
      # Process simulation for this strategy
      sim_key <- paste0("NoQuality_", strategy_name, "_Fixed")
      sim <- simulation_results[[sim_key]]
      
      if (!sim$active) next
      
      if (nrow(bet_data) == 0) {
        # No bets for this event - record zero activity
        event_record <- data.frame(
          Event = event_idx,
          EventID = test_event,
          Field_Quality = event_info_row$Quality,
          Number_of_Players = event_info_row$Number_of_Players,
          Strategy = strategy_name,
          Bankroll_Start = sim$bankroll,
          Num_Bets = 0,
          Total_Stake = 0,
          Total_Winnings = 0,
          Profit_Loss = 0,
          Bankroll_End = sim$bankroll,
          Active = sim$active
        )
        sim$history <- rbind(sim$history, event_record)
        simulation_results[[sim_key]] <- sim
        next
      }
      
      # FIXED STAKE CALCULATION - Changed from percentage to fixed amount
      stake_per_bet <- FIXED_STAKE
      total_stake_needed <- nrow(bet_data) * stake_per_bet
      
      if (sim$bankroll >= total_stake_needed) {
        # Place bets and calculate results using Top20_Profit column
        successful_bets <- sum(bet_data$top_20 == 1)
        # Use Top20_Profit directly - it already accounts for £10 stake and shows net profit
        total_winnings <- sum(ifelse(bet_data$top_20 == 1, bet_data$Top20_Profit, -FIXED_STAKE))
        profit_loss <- total_winnings
        
        # Update simulation state
        sim$bankroll <- sim$bankroll + profit_loss
        sim$total_bets <- sim$total_bets + nrow(bet_data)
        sim$successful_bets <- sim$successful_bets + successful_bets
        sim$total_staked <- sim$total_staked + total_stake_needed
        # For tracking purposes, calculate gross winnings (before deducting stakes)
        gross_winnings <- sum(ifelse(bet_data$top_20 == 1, bet_data$Top20_Profit + FIXED_STAKE, 0))
        sim$total_winnings <- sim$total_winnings + gross_winnings
        
        # Update max bankroll and drawdown
        if (sim$bankroll > sim$max_bankroll) {
          sim$max_bankroll <- sim$bankroll
        }
        current_drawdown <- (sim$max_bankroll - sim$bankroll) / sim$max_bankroll * 100
        if (current_drawdown > sim$max_drawdown) {
          sim$max_drawdown <- current_drawdown
        }
        
        # Check if bankroll hit zero
        if (sim$bankroll <= 0) {
          sim$active <- FALSE
          sim$bankroll <- 0
        }
        
        event_record <- data.frame(
          Event = event_idx,
          EventID = test_event,
          Field_Quality = event_info_row$Quality,
          Number_of_Players = event_info_row$Number_of_Players,
          Strategy = strategy_name,
          Bankroll_Start = sim$bankroll - profit_loss,
          Num_Bets = nrow(bet_data),
          Total_Stake = round(total_stake_needed, 2),
          Total_Winnings = round(gross_winnings, 2),
          Profit_Loss = round(profit_loss, 2),
          Bankroll_End = round(sim$bankroll, 2),
          Active = sim$active
        )
      } else {
        # Insufficient funds - strategy becomes inactive
        sim$active <- FALSE
        
        event_record <- data.frame(
          Event = event_idx,
          EventID = test_event,
          Field_Quality = event_info_row$Quality,
          Number_of_Players = event_info_row$Number_of_Players,
          Strategy = strategy_name,
          Bankroll_Start = sim$bankroll,
          Num_Bets = 0,
          Total_Stake = 0,
          Total_Winnings = 0,
          Profit_Loss = 0,
          Bankroll_End = sim$bankroll,
          Active = FALSE
        )
      }
      
      sim$history <- rbind(sim$history, event_record)
      simulation_results[[sim_key]] <- sim
    }
    
  }, error = function(e) {
    log_message(paste("Error processing event", test_event, ":", e$message), log_file)
  })
}

# Compile and analyze results
log_message("Simulation completed, compiling results...", log_file)

all_history <- data.frame()
final_summary <- data.frame()

for (sim_name in names(simulation_results)) {
  sim <- simulation_results[[sim_name]]
  
  # Add simulation history
  if (nrow(sim$history) > 0) {
    all_history <- rbind(all_history, sim$history)
  }
  
  # Create final summary
  final_bankroll <- ifelse(nrow(sim$history) > 0, tail(sim$history$Bankroll_End, 1), STARTING_BANKROLL)
  total_profit <- final_bankroll - STARTING_BANKROLL
  total_roi <- (total_profit / STARTING_BANKROLL) * 100
  success_rate <- ifelse(sim$total_bets > 0, (sim$successful_bets / sim$total_bets) * 100, 0)
  
  summary_record <- data.frame(
    Strategy = sim$strategy,
    Starting_Bankroll = STARTING_BANKROLL,
    Final_Bankroll = round(final_bankroll, 2),
    Total_Profit = round(total_profit, 2),
    Total_ROI = round(total_roi, 2),
    Total_Bets = sim$total_bets,
    Successful_Bets = sim$successful_bets,
    Success_Rate = round(success_rate, 2),
    Total_Staked = round(sim$total_staked, 2),
    Total_Winnings = round(sim$total_winnings, 2),
    Max_Bankroll = round(sim$max_bankroll, 2),
    Max_Drawdown_Pct = round(sim$max_drawdown, 2),
    Still_Active = sim$active,
    Events_Participated = ifelse(nrow(sim$history) > 0, max(sim$history$Event), 0)
  )
  
  final_summary <- rbind(final_summary, summary_record)
}

# Create simplified Excel report
log_message("Creating Excel report...", log_file)
wb <- createWorkbook()

# Add worksheets
addWorksheet(wb, "Final_Summary")
addWorksheet(wb, "Event_by_Event")
addWorksheet(wb, "Event_Information")
addWorksheet(wb, "Model_Predictions")

# Write data to worksheets
writeData(wb, "Final_Summary", final_summary)
writeData(wb, "Event_by_Event", all_history)
writeData(wb, "Event_Information", event_info)
writeData(wb, "Model_Predictions", all_predictions)

# Add conditional formatting for better visualization
negStyle <- createStyle(fontColour = "#9C0006", bgFill = "#FFC7CE")
posStyle <- createStyle(fontColour = "#006100", bgFill = "#C6EFCE")

# Format Final_Summary ROI column
roi_col <- which(names(final_summary) == "Total_ROI")
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

# Save Excel file
excel_file <- paste0("C:/Projects/Golf/Betting Simulations/Fixed_Stakes_Analysis_", 
                     format(Sys.time(), "%m%d_%H%M"), ".xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)

# ===== RESULTS REPORTING =====
# Top performing strategies
top_strategies <- final_summary %>%
  arrange(desc(Total_ROI)) %>%
  head(10) %>%
  select(Strategy, Total_ROI, Final_Bankroll, Success_Rate, Still_Active)

cat("\nTOP 10 PERFORMING STRATEGIES:\n")
for (i in 1:nrow(top_strategies)) {
  cat(sprintf("%2d. %s: %.1f%% ROI (£%.2f) - %.1f%% success rate %s\n",
              i,
              top_strategies$Strategy[i],
              top_strategies$Total_ROI[i],
              top_strategies$Final_Bankroll[i],
              top_strategies$Success_Rate[i],
              ifelse(top_strategies$Still_Active[i], "✓", "✗")))
}

# Log completion
log_message("Fixed stakes betting simulation completed successfully", log_file)
log_message(paste("Excel report saved to:", excel_file), log_file)

# Display final message
cat(sprintf("\n✓ Analysis complete! Results saved to:\n%s\n\n", excel_file))
