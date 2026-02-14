#### Enhanced Sequential Betting Strategy Simulation with Quality Bin Analysis - PERCENTAGE STAKING ONLY ####
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

# Simulation parameters - moved to top for easy configuration
STARTING_BANKROLL <- 1000
PERCENTAGE_STAKE <- 0.02  # 2% of bankroll per bet
MIN_PLAYERS_PER_EVENT <- 80

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
  
  # Create top 40 target variable
  df$top_40 <- ifelse(df$posn <= 40, 1, 0)
  
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
    resp <- train_data[, "top_40"]
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

# Enhanced results analysis function
analyze_results <- function(simulation_results, final_summary, log_file = NULL) {
  log_message("Analyzing simulation results...", log_file)
  
  # Create quality bin performance analysis
  quality_bin_analysis <- final_summary %>%
    filter(Approach %in% c("Quality3Bin", "Quality4Bin")) %>%
    group_by(Approach, Quality_Filter, Strategy) %>%
    summarize(
      Avg_ROI = mean(Total_ROI, na.rm = TRUE),
      Avg_Success_Rate = mean(Success_Rate, na.rm = TRUE),
      Total_Events = sum(Events_Participated, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(Approach, Quality_Filter, desc(Avg_ROI))
  
  # Find best strategy for each quality bin
  best_by_quality <- quality_bin_analysis %>%
    group_by(Approach, Quality_Filter) %>%
    slice_max(Avg_ROI, n = 1) %>%
    ungroup()
  
  # Create approach comparison
  approach_comparison <- final_summary %>%
    select(Approach, Strategy, Final_Bankroll, Total_ROI, Success_Rate, Max_Drawdown_Pct, Still_Active) %>%
    arrange(Strategy, desc(Total_ROI))
  
  return(list(
    quality_bin_analysis = quality_bin_analysis,
    best_by_quality = best_by_quality,
    approach_comparison = approach_comparison
  ))
}

# ===== MAIN EXECUTION =====
# Create log file
log_file <- paste0("C:/Projects/Golf/temp_status/betting_simulation_percentage_", format(Sys.time(), "%m%d_%H%M"), ".log")
log_message("Starting Enhanced Sequential Betting Simulation with Quality Analysis - PERCENTAGE STAKING ONLY", log_file)

# Load and validate data
log_message("Loading training data...", log_file)
df_old <- read_excel("./Data/PGA_revised_odds.xlsx")
names(df_old) <- gsub("^_", "X_", names(df_old))
df_old$rating <- as.numeric(df_old$rating)
df_old[is.na(df_old$Top40_Odds), "Top40_Odds"] <- 0
df_old[is.na(df_old$Top20_Odds), "Top20_Odds"] <- 0
df_old <- df_old[complete.cases(df_old), ]
df_old <- validate_data(df_old, "training")

log_message("Loading test data...", log_file)
df_new <- read.csv("./Data/PGA_update_2805.csv")
df_new$rating <- as.numeric(df_new$rating)
df_new[is.na(df_new$Top40_Odds), "Top40_Odds"] <- 0
df_new[is.na(df_new$Top20_Odds), "Top20_Odds"] <- 0
df_new <- df_new[, c(1:70, 72)]
df_new <- df_new[complete.cases(df_new), ]
df_new <- validate_data(df_new, "test")

# Load event quality data
log_message("Loading event quality data...", log_file)
eventQuality <- read.csv("./Data/FieldQuality.csv")
df_old <- df_old %>% left_join(eventQuality, by = c("eventID" = "EventID"))

# Create derived features
df_old <- create_derived_features(df_old, log_file)
df_new <- create_derived_features(df_new, log_file)

# Define model variables
model_vars <- c("log_rating", "current_rank", "compat_rank", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "sgapp", "Starts_Not10", "Top40_Odds")

# Check available variables
available_vars_old <- model_vars[model_vars %in% names(df_old)]
available_vars_new <- model_vars[model_vars %in% names(df_new)]
available_vars <- intersect(available_vars_old, available_vars_new)

log_message(paste("Available model variables:", length(available_vars), "of", length(model_vars)), log_file)

# Create datasets
train_data <- df_old %>%
  select(playerID, eventID, posn, top_40, all_of(available_vars))

test_data <- df_new %>%
  select(playerID, eventID, posn, top_40, Quality, all_of(available_vars))

log_message(paste("Training data:", nrow(train_data), "observations from", 
                  length(unique(train_data$eventID)), "events"), log_file)
log_message(paste("Test data:", nrow(test_data), "observations from", 
                  length(unique(test_data$eventID)), "events"), log_file)

# Generate betting strategies
strategies <- generate_betting_strategies(log_file = log_file)

# Train ensemble model
myBiomodEM <- train_ensemble_model(train_data, available_vars, log_file)

# Create event quality mapping and initialize simulations
new_events <- unique(test_data$eventID)
event_quality_map <- test_data %>%
  select(eventID, Quality) %>%
  distinct() %>%
  left_join(
    test_data %>%
      group_by(eventID) %>%
      summarize(Number_of_Players = n(), .groups = "drop"),
    by = "eventID"
  ) %>%
  mutate(
    Quality_3Bin = case_when(
      Quality < 70 ~ "Low_<70",
      Quality >= 70 & Quality < 72 ~ "Med_70-72",
      Quality >= 72 ~ "High_72+",
      TRUE ~ "Unknown"
    ),
    Quality_4Bin = case_when(
      Quality < 70 ~ "Low_<70",
      Quality >= 70 & Quality < 71 ~ "Med_70-71",
      Quality >= 71 & Quality < 72 ~ "High_71-72",
      Quality >= 72 ~ "VHigh_72+",
      TRUE ~ "Unknown"
    )
  )

log_message(paste("Quality 3-bin distribution:", paste(table(event_quality_map$Quality_3Bin), collapse = ", ")), log_file)
log_message(paste("Quality 4-bin distribution:", paste(table(event_quality_map$Quality_4Bin), collapse = ", ")), log_file)

# Initialize simulation tracking
simulation_results <- list()
approach_names <- c("NoQuality", "Quality3Bin", "Quality4Bin")

# Create simulation configurations - PERCENTAGE STAKING ONLY
for (approach in approach_names) {
  for (strategy_name in names(strategies)) {
    if (approach == "NoQuality") {
      sim_key <- paste0(approach, "_", strategy_name, "_Percentage")
      simulation_results[[sim_key]] <- list(
        approach = approach,
        strategy = strategy_name,
        type = "Percentage",
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
    } else {
      quality_bins <- if (approach == "Quality3Bin") {
        c("Low_<70", "Med_70-72", "High_72+")
      } else {
        c("Low_<70", "Med_70-71", "High_71-72", "VHigh_72+")
      }
      
      for (quality_bin in quality_bins) {
        sim_key <- paste0(approach, "_", quality_bin, "_", strategy_name, "_Percentage")
        simulation_results[[sim_key]] <- list(
          approach = approach,
          strategy = strategy_name,
          type = "Percentage",
          quality_filter = quality_bin,
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
    }
  }
}

log_message(paste("Total simulations to run:", length(simulation_results)), log_file)

# Run event-by-event simulation
log_message(paste("Running simulation on", length(new_events), "events in sequence"), log_file)

for (event_idx in 1:length(new_events)) {
  test_event <- new_events[event_idx]
  
  if (event_idx %% 5 == 0) {
    log_message(paste("Processing event", event_idx, "of", length(new_events), ":", test_event), log_file, console = FALSE)
  }
  
  # Get event data and quality info
  event_data <- test_data %>%
    filter(eventID == test_event) %>%
    mutate(Top40_Odds = filter(df_new, eventID == test_event)$Top40_Odds)
  
  event_quality_info <- event_quality_map[event_quality_map$eventID == test_event, ]
  
  if (nrow(event_data) < MIN_PLAYERS_PER_EVENT) {
    log_message(paste("Skipping event", test_event, "- insufficient players (", nrow(event_data), ")"), log_file, console = FALSE)
    next
  }
  
  quality_3bin <- event_quality_info$Quality_3Bin
  quality_4bin <- event_quality_info$Quality_4Bin
  
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
      event_data[, c("eventID", "playerID", "Top40_Odds", "posn", "top_40")],
      by = c("eventID", "playerID"), all.x = TRUE
    )
    
    PGA_Prediction$Valid_Odds <- PGA_Prediction$Top40_Odds > 1
    
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
        PGA_Prediction$rank_order <- rank(-PGA_Prediction$Model_Score, ties.method = "random")
        bet_players <- PGA_Prediction$rank_order >= actual_start_rank & 
          PGA_Prediction$rank_order <= actual_end_rank & 
          PGA_Prediction$Valid_Odds
      }
      
      bet_data <- PGA_Prediction[bet_players, ]
      
      # Process all relevant simulations
      for (sim_key in names(simulation_results)) {
        sim <- simulation_results[[sim_key]]
        
        if (!sim$active) next
        
        # Check if this simulation should process this event
        should_process <- FALSE
        
        if (sim$approach == "NoQuality") {
          should_process <- grepl(paste0("NoQuality_", strategy_name, "_Percentage"), sim_key)
        } else if (sim$approach == "Quality3Bin") {
          should_process <- (sim$quality_filter == quality_3bin && 
                               grepl(paste0("Quality3Bin_", sim$quality_filter, "_", strategy_name, "_Percentage"), sim_key))
        } else if (sim$approach == "Quality4Bin") {
          should_process <- (sim$quality_filter == quality_4bin && 
                               grepl(paste0("Quality4Bin_", sim$quality_filter, "_", strategy_name, "_Percentage"), sim_key))
        }
        
        if (!should_process) next
        
        if (nrow(bet_data) == 0) {
          # No bets for this event - record zero activity
          event_record <- data.frame(
            Event = event_idx,
            EventID = test_event,
            Field_Quality = event_quality_info$Quality,
            Number_of_Players = event_quality_info$Number_of_Players,
            Approach = sim$approach,
            Strategy = strategy_name,
            Quality_Filter = sim$quality_filter,
            Quality_Bin = ifelse(sim$approach == "NoQuality", "All", 
                                 ifelse(sim$approach == "Quality3Bin", quality_3bin, quality_4bin)),
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
        
        # Calculate percentage stake amounts
        stake_per_bet <- sim$bankroll * PERCENTAGE_STAKE
        total_stake_needed <- nrow(bet_data) * stake_per_bet
        
        if (sim$bankroll >= total_stake_needed) {
          # Place bets and calculate results
          successful_bets <- sum(bet_data$top_40 == 1)
          total_winnings <- sum(ifelse(bet_data$top_40 == 1, stake_per_bet * bet_data$Top40_Odds, 0))
          profit_loss <- total_winnings - total_stake_needed
          
          # Update simulation state
          sim$bankroll <- sim$bankroll + profit_loss
          sim$total_bets <- sim$total_bets + nrow(bet_data)
          sim$successful_bets <- sim$successful_bets + successful_bets
          sim$total_staked <- sim$total_staked + total_stake_needed
          sim$total_winnings <- sim$total_winnings + total_winnings
          
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
            Field_Quality = event_quality_info$Quality,
            Number_of_Players = event_quality_info$Number_of_Players,
            Approach = sim$approach,
            Strategy = strategy_name,
            Quality_Filter = sim$quality_filter,
            Quality_Bin = ifelse(sim$approach == "NoQuality", "All", 
                                 ifelse(sim$approach == "Quality3Bin", quality_3bin, quality_4bin)),
            Bankroll_Start = sim$bankroll - profit_loss,
            Num_Bets = nrow(bet_data),
            Total_Stake = round(total_stake_needed, 2),
            Total_Winnings = round(total_winnings, 2),
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
            Field_Quality = event_quality_info$Quality,
            Number_of_Players = event_quality_info$Number_of_Players,
            Approach = sim$approach,
            Strategy = strategy_name,
            Quality_Filter = sim$quality_filter,
            Quality_Bin = ifelse(sim$approach == "NoQuality", "All", 
                                 ifelse(sim$approach == "Quality3Bin", quality_3bin, quality_4bin)),
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
    Approach = sim$approach,
    Strategy = sim$strategy,
    Quality_Filter = sim$quality_filter,
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

# Analyze results
analysis_results <- analyze_results(simulation_results, final_summary, log_file)

# Create comprehensive Excel report
log_message("Creating comprehensive Excel report...", log_file)
wb <- createWorkbook()

# Add worksheets with enhanced formatting
addWorksheet(wb, "Final_Summary")
addWorksheet(wb, "Quality_Bin_Performance")
addWorksheet(wb, "Event_by_Event")
addWorksheet(wb, "Event_Quality_Map")

# Write data to worksheets
writeData(wb, "Final_Summary", final_summary)
writeData(wb, "Quality_Bin_Performance", analysis_results$quality_bin_analysis)
writeData(wb, "Event_by_Event", all_history)
writeData(wb, "Event_Quality_Map", event_quality_map)

# Add conditional formatting for better visualization
# Highlight positive ROI in green, negative in red
negStyle <- createStyle(fontColour = "#9C0006", bgFill = "#FFC7CE")
posStyle <- createStyle(fontColour = "#006100", bgFill = "#C6EFCE")

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
excel_file <- paste0("C:/Projects/Golf/Betting Simulations/Percentage_Only_Analysis_", 
                     format(Sys.time(), "%m%d_%H%M"), ".xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)

# ===== EXTRA RESULTS REPORTING =====
# Top performing strategies overall
top_strategies <- final_summary %>%
  arrange(desc(Total_ROI)) %>%
  head(10) %>%
  select(Approach, Strategy, Quality_Filter, Total_ROI, Final_Bankroll, Success_Rate, Still_Active)

cat("\nTOP 10 PERFORMING STRATEGIES (All Approaches):\n")
for (i in 1:nrow(top_strategies)) {
  cat(sprintf("%2d. %s-%s [%s]: %.1f%% ROI (£%.2f) - %.1f%% success rate %s\n",
              i,
              top_strategies$Approach[i],
              top_strategies$Strategy[i],
              top_strategies$Quality_Filter[i],
              top_strategies$Total_ROI[i],
              top_strategies$Final_Bankroll[i],
              top_strategies$Success_Rate[i],
              ifelse(top_strategies$Still_Active[i], "✓", "✗")))
}

# Log completion
log_message("Enhanced quality bin analysis completed successfully", log_file)
log_message(paste("Excel report saved to:", excel_file), log_file)

# Display final message
cat(sprintf("\n✓ Analysis complete! Results saved to:\n%s\n\n", excel_file))