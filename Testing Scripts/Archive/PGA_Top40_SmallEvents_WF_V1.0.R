# ===== CONFIGURATION SECTION =====
# Set working directory and parameters
setwd("C:/Projects/Golf")
set.seed(42)

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.7
MODEL_NAME <- paste0("Top20_WF_SmallEvents_", format(Sys.time(), "%d%m_%H%M"))
MAX_PLAYERS_PER_EVENT <- 119

# Walk-forward parameters
RETRAIN_FREQUENCY <- 1  # Retrain after every N events
MIN_TRAINING_EVENTS <- 20  # Minimum events before first prediction

# Calibration parameters
ENABLE_CALIBRATION <- TRUE

# Betting simulation parameters
ENABLE_BETTING_SIMULATION <- TRUE

# ===== HELPER FUNCTIONS =====
# Model training function
train_ensemble_model <- function(train_data, model_vars, log_file = NULL) {
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
    
  }, error = function(e) {
    stop(e)
  })
}

# Kelly betting simulation function
simulate_kelly_betting_custom <- function(df, 
                                          initial_bankroll = 1000,
                                          kelly_multiplier = 0.5,
                                          max_bet_amount = 50,
                                          max_bet_pct = 0.05,
                                          max_event_pct = 0.20,
                                          min_kelly_threshold = 0.02,
                                          min_expected_value = 0.05,
                                          min_prob_edge = 0.03,
                                          max_bets_per_event = 5,
                                          use_dynamic_bankroll = FALSE,
                                          min_odds = 1,
                                          max_odds = 1000) { 
  
  # Sort by EventNumber for chronological betting
  df_sorted <- df %>% arrange(EventNumber, PlayerID)
  
  # Get unique events in chronological order
  unique_events <- unique(df_sorted$EventNumber)
  
  # Initialize tracking variables
  bankroll <- initial_bankroll
  bet_history <- data.frame()
  
  # Track bankroll over time
  bankroll_history <- data.frame(
    event_number = 0,
    EventNumber = NA,
    EventID = NA,
    bankroll_start = initial_bankroll,
    bankroll_end = initial_bankroll,
    event_profit = 0,
    cumulative_profit = 0,
    bets_in_event = 0,
    total_bet_amount = 0,
    stringsAsFactors = FALSE
  )
  
  total_bet_counter <- 0
  
  # Process each event
  for(event_idx in 1:length(unique_events)) {
    current_event_num <- unique_events[event_idx]
    event_data <- df_sorted[df_sorted$EventNumber == current_event_num, ]
    
    if (nrow(event_data) == 0) next
    
    current_event_id <- event_data$EventID[1]
    
    # Store bankroll at start of event
    event_start_bankroll <- bankroll
    bet_counter_for_event <- 0
    
    # Calculate potential bets for this event
    potential_bets <- data.frame()
    
    for(i in 1:nrow(event_data)) {
      row <- event_data[i, ]
      
      # Skip if missing required data
      if (is.na(row$Calibrated_Probability) || is.na(row$Top20_Odds) || 
          row$Top20_Odds <= 0 || !row$Valid_Odds) {
        next
      }
      
      # Calculate probability edge over market
      implied_prob <- 1 / row$Top20_Odds
      prob_edge <- row$Calibrated_Probability - implied_prob
      
      # Use pre-calculated Kelly fraction, but cap at 100%
      kelly_frac <- pmax(0, pmin(row$Kelly_Fraction, 1))
      
      # Apply Kelly multiplier
      adjusted_kelly <- kelly_frac * kelly_multiplier
      
      # Calculate reference bankroll
      reference_bankroll <- ifelse(use_dynamic_bankroll, event_start_bankroll, initial_bankroll)
      
      # Apply all filters
      passes_filters <- (
        adjusted_kelly > min_kelly_threshold &&
          row$Expected_Value > min_expected_value &&
          prob_edge > min_prob_edge &&
          reference_bankroll > max_bet_amount &&
          row$Top20_Odds >= min_odds &&
          row$Top20_Odds <= max_odds
      )
      
      if(passes_filters) {
        # Calculate bet amounts
        theoretical_bet <- reference_bankroll * adjusted_kelly
        max_bankroll_bet <- reference_bankroll * max_bet_pct
        constrained_bet <- min(theoretical_bet, max_bet_amount, max_bankroll_bet)
        
        # Store potential bet
        potential_bet <- data.frame(
          row_index = i,
          PlayerID = row$PlayerID,
          theoretical_bet = theoretical_bet,
          constrained_bet = constrained_bet,
          adjusted_kelly = adjusted_kelly,
          original_kelly = kelly_frac,
          kelly_multiplier_used = kelly_multiplier,
          odds = row$Top20_Odds,
          probability = row$Calibrated_Probability,
          expected_value = row$Expected_Value,
          prob_edge = prob_edge,
          implied_prob = implied_prob,
          bet_capped_amount = constrained_bet < theoretical_bet && theoretical_bet > max_bet_amount,
          bet_capped_pct = constrained_bet < theoretical_bet && theoretical_bet > max_bankroll_bet,
          stringsAsFactors = FALSE
        )
        
        potential_bets <- rbind(potential_bets, potential_bet)
      }
    }
    
    # Apply maximum bets per event filter
    if(nrow(potential_bets) > max_bets_per_event) {
      potential_bets <- potential_bets %>%
        arrange(desc(expected_value)) %>%
        slice_head(n = max_bets_per_event)
    }
    
    # Apply event budget constraint
    if(nrow(potential_bets) > 0) {
      total_event_allocation <- sum(potential_bets$constrained_bet)
      max_event_budget <- reference_bankroll * max_event_pct
      
      # Scale down if exceeding event budget
      if(total_event_allocation > max_event_budget) {
        scaling_factor <- max_event_budget / total_event_allocation
        potential_bets$final_bet_amount <- round(potential_bets$constrained_bet * scaling_factor, 0) # Rounded to full number
        potential_bets$event_scaled <- TRUE
      } else {
        potential_bets$final_bet_amount <- round(potential_bets$constrained_bet, 0) # Rounded to full number
        potential_bets$event_scaled <- FALSE
      }
      
      # Place bets and resolve outcomes
      event_total_bet <- 0
      event_bets <- data.frame()
      
      for(j in 1:nrow(potential_bets)) {
        bet_counter_for_event <- bet_counter_for_event + 1
        total_bet_counter <- total_bet_counter + 1
        
        bet_data <- potential_bets[j, ]
        original_row <- event_data[bet_data$row_index, ]
        
        event_total_bet <- event_total_bet + bet_data$final_bet_amount
        
        # Determine outcome and calculate profit
        won_bet <- original_row$Actual_Top20 == 1
        
        if(won_bet) {
          profit <- (bet_data$final_bet_amount / 10) * original_row$Top20_Profit
        } else {
          profit <- -bet_data$final_bet_amount
        }
        
        # Record bet
        bet_record <- data.frame(
          bet_number = total_bet_counter,
          event_number = event_idx,
          EventNumber = current_event_num,
          EventID = current_event_id,
          PlayerID = bet_data$PlayerID,
          bet_amount = bet_data$final_bet_amount,
          theoretical_bet = bet_data$theoretical_bet,
          constrained_bet = bet_data$constrained_bet,
          final_bet_fraction = bet_data$final_bet_amount / event_start_bankroll,
          adjusted_kelly = bet_data$adjusted_kelly,
          original_kelly = bet_data$original_kelly,
          kelly_multiplier = bet_data$kelly_multiplier_used,
          odds = bet_data$odds,
          probability = bet_data$probability,
          expected_value = bet_data$expected_value,
          prob_edge = bet_data$prob_edge,
          implied_prob = bet_data$implied_prob,
          bet_capped_amount = bet_data$bet_capped_amount,
          bet_capped_pct = bet_data$bet_capped_pct,
          event_scaled = bet_data$event_scaled,
          bankroll_at_bet = event_start_bankroll,
          won = won_bet,
          profit = profit,
          stringsAsFactors = FALSE
        )
        
        event_bets <- rbind(event_bets, bet_record)
      }
      
      # Update bankroll and tracking
      total_event_profit <- sum(event_bets$profit)
      bankroll <- bankroll + total_event_profit
      bet_history <- rbind(bet_history, event_bets)
      
      # Record event summary
      event_summary <- data.frame(
        event_number = event_idx,
        EventNumber = current_event_num,
        EventID = current_event_id,
        bankroll_start = event_start_bankroll,
        bankroll_end = bankroll,
        event_profit = total_event_profit,
        cumulative_profit = bankroll - initial_bankroll,
        bets_in_event = bet_counter_for_event,
        total_bet_amount = event_total_bet,
        stringsAsFactors = FALSE
      )
      
    } else {
      # No bets placed
      event_summary <- data.frame(
        event_number = event_idx,
        EventNumber = current_event_num,
        EventID = current_event_id,
        bankroll_start = event_start_bankroll,
        bankroll_end = bankroll,
        event_profit = 0,
        cumulative_profit = bankroll - initial_bankroll,
        bets_in_event = 0,
        total_bet_amount = 0,
        stringsAsFactors = FALSE
      )
    }
    
    bankroll_history <- rbind(bankroll_history, event_summary)
  }
  
  # Calculate summary statistics
  if(nrow(bet_history) > 0) {
    win_rate <- mean(bet_history$won, na.rm = TRUE)
    avg_bet_size <- mean(bet_history$bet_amount, na.rm = TRUE)
    total_wagered <- sum(bet_history$bet_amount, na.rm = TRUE)
    avg_odds <- mean(bet_history$odds, na.rm = TRUE)
    avg_prob_edge <- mean(bet_history$prob_edge, na.rm = TRUE)
  } else {
    win_rate <- avg_bet_size <- total_wagered <- avg_odds <- avg_prob_edge <- 0
  }
  
  return(list(
    bet_history = bet_history,
    bankroll_history = bankroll_history,
    final_bankroll = bankroll,
    total_profit = bankroll - initial_bankroll,
    roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100,
    total_bets = total_bet_counter,
    win_rate = win_rate,
    avg_bet_size = avg_bet_size,
    total_wagered = total_wagered,
    avg_odds = avg_odds,
    avg_prob_edge = avg_prob_edge,
    parameters = list(
      initial_bankroll = initial_bankroll,
      kelly_multiplier = kelly_multiplier,
      max_bet_amount = max_bet_amount,
      max_bet_pct = max_bet_pct,
      max_event_pct = max_event_pct,
      min_kelly_threshold = min_kelly_threshold,
      min_expected_value = min_expected_value,
      min_prob_edge = min_prob_edge,
      max_bets_per_event = max_bets_per_event,
      use_dynamic_bankroll = use_dynamic_bankroll
    )
  ))
}

# Function to print simulation results
print_simulation_results <- function(sim_result) {
  params <- sim_result$parameters
  
  cat("Kelly Betting Simulation Results\n")
  cat("================================\n")
  cat("Parameters:\n")
  cat("  Kelly Multiplier:", params$kelly_multiplier, 
      paste0("(", ifelse(params$kelly_multiplier == 1, "Full", 
                         ifelse(params$kelly_multiplier == 0.5, "Half", 
                                paste0(params$kelly_multiplier * 100, "%"))), " Kelly)\n"))
  cat("  Initial Bankroll: £", params$initial_bankroll, "\n")
  cat("  Max Bet Amount: £", params$max_bet_amount, "\n")
  cat("  Max Bet %:", params$max_bet_pct * 100, "%\n")
  cat("  Max Event %:", params$max_event_pct * 100, "%\n")
  cat("  Min Kelly Threshold:", params$min_kelly_threshold * 100, "%\n")
  cat("  Min Expected Value:", params$min_expected_value * 100, "%\n")
  cat("  Min Probability Edge:", params$min_prob_edge * 100, "%\n")
  cat("  Max Bets Per Event:", params$max_bets_per_event, "\n")
  cat("  Dynamic Bankroll:", params$use_dynamic_bankroll, "\n\n")
  
  cat("Results:\n")
  cat("  Final Bankroll: £", round(sim_result$final_bankroll, 2), "\n")
  cat("  Total Profit: £", round(sim_result$total_profit, 2), "\n")
  cat("  ROI:", round(sim_result$roi, 2), "%\n")
  cat("  Total Bets:", sim_result$total_bets, "\n")
  
  if(sim_result$total_bets > 0) {
    cat("  Win Rate:", round(sim_result$win_rate * 100, 2), "%\n")
    cat("  Average Bet Size: £", round(sim_result$avg_bet_size, 2), "\n")
    cat("  Total Wagered: £", round(sim_result$total_wagered, 2), "\n")
    cat("  Average Odds:", round(sim_result$avg_odds, 2), "\n")
    cat("  Average Probability Edge:", round(sim_result$avg_prob_edge * 100, 2), "%\n")
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
eventQuality <- read.csv("./Data/FieldQuality.csv")
eventDates <- read.csv("./Data/EventDates.csv")

df_all <- read_excel("./Data/PGA_processed_features.xlsx")
df_all <- df_all[complete.cases(df_all),]

fs <- df_all %>%
  group_by(eventID) %>%
  summarise(N = n()) %>%
  mutate(Below_Max = ifelse(N > 119, 0, 1)) %>%
  filter(Below_Max == 1)

# Create sequential event numbering based on dates
df_all <- df_all %>%
  filter(eventID %in% fs$eventID) %>%
  mutate(Date = dmy(Date)) %>%
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

cat("Total events to process:", total_events, "\n")
cat("Date range:", as.character(min(unique_events$event_date)), "to", as.character(max(unique_events$event_date)), "\n")

# Define model variables
model_vars <- c("rating_vs_field_best",
                "yr3_All",
                "Top40_Odds",
                "Top5_rank",
                "compat2",
                "sgp_field_zscore",
                "sgtee_field_zscore",
                "course",
                "sgatg_vs_field_median", 
                "Quality", 
                "sgapp_vs_field_median", 
                "location", 
                "field")

# Check available variables
missing_vars <- model_vars[model_vars %not in% names(df_all)]
available_vars <- model_vars[model_vars %in% names(df_all)]

cat("Available variables:", length(available_vars), "/", length(model_vars), "\n")
if (length(missing_vars) > 0) {
  cat("Missing variables:", paste(missing_vars, collapse = ", "), "\n")
}

# ===== WALK-FORWARD VALIDATION WITH CALIBRATION =====
# Initialize tracking
all_predictions <- data.frame()
calibration_history <- data.frame()  # Initialize this
trained_models <- list()
trained_calibration_models <- list()
last_training_event <- 0

# Walk-forward loop
for (current_event in (MIN_TRAINING_EVENTS + 1):total_events) {
  
  current_event_id <- unique_events$eventID[unique_events$event_number == current_event]
  cat("Processing event", current_event, "of", total_events, ":", current_event_id, "\n")
  
  # Check if we need to retrain
  need_retrain <- (current_event - last_training_event) >= RETRAIN_FREQUENCY
  
  if (need_retrain || length(trained_models) == 0) {
    cat("  Retraining model and calibration...\n")
    
    # Training data: all events before current event
    train_data <- df_all %>%
      filter(event_number < current_event) %>%
      select(playerID, eventID, posn, top_20, all_of(available_vars))
    
    # Train ensemble model
    current_model <- train_ensemble_model(train_data, available_vars)
    trained_models[[as.character(current_event)]] <- current_model
    
    # Train calibration model using all available historical predictions
    # Initialize calibration_model
    calibration_model <- NULL
    
    # Train calibration model using predictions from the ensemble model just trained
    if (ENABLE_CALIBRATION) {
      cat("  Training calibration model with", nrow(train_data), "training predictions...\n")
      
      tryCatch({
        train_data <- train_data %>%
          left_join(select(df_all, playerID, eventID, Top20_Odds), by = c("playerID", "eventID"))
        
        # Get predictions from the ensemble model on training data
        logit_model_data <- data.frame(
          Model_Score = current_model@models.prediction@val$pred,
          Actual_Top20 = train_data$top_20,
          Top20_Odds = train_data$Top20_Odds
        )
        
        # Train calibration model
        temp_calibration_model <- glm(Actual_Top20 ~ Model_Score + Top20_Odds,
                                      data = logit_model_data,
                                      family = binomial())
        
        # Check for convergence and store the model
        if (!temp_calibration_model$converged) {
          cat("    Warning: Calibration model did not converge\n")
          trained_calibration_models[[as.character(current_event)]] <- NULL
        } else {
          trained_calibration_models[[as.character(current_event)]] <- temp_calibration_model
          calibration_model <- temp_calibration_model
          cat("    Calibration model trained successfully\n")
        }
        
      }, error = function(e) {
        cat("  Warning: Could not train calibration model:", e$message, "\n")
        trained_calibration_models[[as.character(current_event)]] <- NULL
      })
    } else {
      cat("  Insufficient training data for calibration model (", nrow(train_data), "samples)\n")
      trained_calibration_models[[as.character(current_event)]] <- NULL
    }
    
    last_training_event <- current_event
    
  } else {
    # Use most recent models
    model_keys <- as.numeric(names(trained_models))
    latest_key <- max(model_keys)
    current_model <- trained_models[[as.character(latest_key)]]
    
    # Use most recent calibration model
    calibration_model <- NULL
    if (length(trained_calibration_models) > 0) {
      cal_keys <- as.numeric(names(trained_calibration_models))
      cal_keys <- cal_keys[!is.na(cal_keys)]
      if (length(cal_keys) > 0) {
        latest_cal_key <- max(cal_keys)
        calibration_model <- trained_calibration_models[[as.character(latest_cal_key)]]
      }
    }
    
    cat("  Using model from event", latest_key, "\n")
    if (!is.null(calibration_model)) {
      cat("  Using calibration model from event", latest_cal_key, "\n")
    } else {
      cat("  No calibration model available\n")
    }
  }
  
  # Test data: current event only
  test_columns <- c("playerID", "eventID", "posn", "top_20", "Quality", 
                    "Top20_Odds", "Top20_Profit", "rating", available_vars)
  test_columns <- unique(test_columns)
  available_test_columns <- test_columns[test_columns %in% names(df_all)]
  
  test_data <- df_all %>%
    filter(event_number == current_event) %>%
    select(all_of(available_test_columns))
  
  # Ensure required columns exist
  if (!"Top20_Odds" %in% names(test_data)) test_data$Top20_Odds <- NA
  if (!"Top20_Profit" %in% names(test_data)) test_data$Top20_Profit <- NA  
  if (!"rating" %in% names(test_data)) test_data$rating <- NA
  
  # Skip if insufficient players
  if (nrow(test_data) > MAX_PLAYERS_PER_EVENT) {
    cat("  Skipping - insufficient players:", nrow(test_data), "\n")
    next
  }
  
  # Make predictions for this event
  tryCatch({
    cat("  Making predictions for", nrow(test_data), "players...\n")
    
    # Validate test data
    missing_vars_test <- available_vars[!available_vars %in% names(test_data)]
    if (length(missing_vars_test) > 0) {
      cat("  ERROR: Missing variables in test data:", paste(missing_vars_test, collapse = ", "), "\n")
      next
    }
    
    test_expl <- test_data[, available_vars]
    test_resp.xy <- test_data[, c("playerID", "eventID")]
    
    # Check for NA values
    if (any(is.na(test_expl))) {
      cat("  WARNING: NA values found in explanatory variables\n")
    }
    
    # Run biomod projection
    myBiomodProj <- BIOMOD_EnsembleForecasting(
      bm.em = current_model,
      bm.proj = NULL,
      proj.name = paste0("event_", current_event_id),
      new.env = test_expl,
      new.env.xy = test_resp.xy,
      models.chosen = "all",
      metric.binary = "TSS",
      metric.filter = "TSS",
      na.rm = TRUE
    )
    
    # Process predictions
    PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
    colnames(PGA_Prediction)[3] <- "Model_Score"
    
    # Merge with test data
    essential_cols <- c("eventID", "playerID", "posn", "top_20")
    merge_cols <- intersect(names(test_data), c(essential_cols, "Top20_Odds", "Top20_Profit", "rating", "Quality"))
    
    PGA_Prediction <- merge(
      PGA_Prediction,
      test_data[, merge_cols],
      by = c("eventID", "playerID"), all.x = TRUE
    )
    
    # Handle missing values
    PGA_Prediction$Top20_Odds <- ifelse(is.na(PGA_Prediction$Top20_Odds), 0, PGA_Prediction$Top20_Odds)
    PGA_Prediction$Top20_Profit <- ifelse(is.na(PGA_Prediction$Top20_Profit), 0, PGA_Prediction$Top20_Profit)
    PGA_Prediction$rating <- ifelse(is.na(PGA_Prediction$rating), 0, PGA_Prediction$rating)
    
    # Apply calibration
    if (!is.null(calibration_model) && ENABLE_CALIBRATION) {
      cat("  Applying calibration...\n")
      
      calibration_data <- data.frame(
        Model_Score = PGA_Prediction$Model_Score,
        Top20_Odds = PGA_Prediction$Top20_Odds
      )
      
      # Handle missing odds for calibration
      calibration_data$Top20_Odds[is.na(calibration_data$Top20_Odds)] <- 
        mean(calibration_data$Top20_Odds, na.rm = TRUE)
      
      # Debug: Check calibration inputs
      cat("    Model Score range:", round(min(calibration_data$Model_Score), 3), "to", round(max(calibration_data$Model_Score), 3), "\n")
      cat("    Odds range:", round(min(calibration_data$Top20_Odds), 3), "to", round(max(calibration_data$Top20_Odds), 3), "\n")
      
      PGA_Prediction$Calibrated_Probability <- predict(calibration_model, 
                                                       calibration_data, 
                                                       type = "response")
      
      # Debug: Check calibration outputs
      cat("    Calibrated prob range:", round(min(PGA_Prediction$Calibrated_Probability), 3), "to", round(max(PGA_Prediction$Calibrated_Probability), 3), "\n")
      
      # Safety check for extreme probabilities
      extreme_probs <- sum(PGA_Prediction$Calibrated_Probability >= 0.99 | PGA_Prediction$Calibrated_Probability <= 0.01)
      if (extreme_probs > nrow(PGA_Prediction) * 0.5) {
        cat("    WARNING: Many extreme probabilities detected (", extreme_probs, "/", nrow(PGA_Prediction), ") - using fallback\n")
        PGA_Prediction$Calibrated_Probability <- plogis(PGA_Prediction$Model_Score)
      }
      
    } else {
      # Fallback: use logistic transformation of raw scores
      cat("  Using fallback calibration (logistic transformation)\n")
      PGA_Prediction$Calibrated_Probability <- plogis(PGA_Prediction$Model_Score)
      cat("    Fallback prob range:", round(min(PGA_Prediction$Calibrated_Probability), 3), "to", round(max(PGA_Prediction$Calibrated_Probability), 3), "\n")
    }
    
    # Calculate derived metrics
    PGA_Prediction$Valid_Odds <- PGA_Prediction$Top20_Odds > 1
    PGA_Prediction$Model_Rank <- rank(-PGA_Prediction$Model_Score, ties.method = "first")
    PGA_Prediction$Prediction_Correct <- ifelse(
      (PGA_Prediction$top_20 == 1 & PGA_Prediction$Model_Rank <= 20) | 
        (PGA_Prediction$top_20 == 0 & PGA_Prediction$Model_Rank > 20), 
      1, 0)
    
    PGA_Prediction$Top20_Winnings <- ifelse(PGA_Prediction$top_20 == 1, PGA_Prediction$Top20_Profit, -10)
    
    # Add Kelly calculations
    PGA_Prediction <- PGA_Prediction %>%
      mutate(
        # Odds are already in decimal format
        decimal_odds = Top20_Odds,
        b = pmax(0, decimal_odds - 1),  # Net odds (protect against odds <= 1)
        p = Calibrated_Probability,
        q = 1 - p,
        
        # Calculate Kelly fraction: (bp - q) / b, handle division by zero
        Kelly_Fraction = ifelse(b > 0, pmax(0, (b * p - q) / b), 0),
        
        # Expected value
        Expected_Value = ifelse(decimal_odds > 0, (p * (decimal_odds - 1)) - ((1 - p) * 1), 0),
        
        # Market analysis
        Implied_Probability = ifelse(decimal_odds > 0, 1 / decimal_odds, 0),
        Prob_Edge = p - Implied_Probability
      )
    
    # Store results
    prediction_record <- data.frame(
      EventNumber = current_event,
      EventID = PGA_Prediction$eventID,
      PlayerID = PGA_Prediction$playerID,
      Model_Score = round(PGA_Prediction$Model_Score, 4),
      Model_Rank = PGA_Prediction$Model_Rank,
      Actual_Top20 = PGA_Prediction$top_20,
      Actual_Position = PGA_Prediction$posn,
      Prediction_Correct = PGA_Prediction$Prediction_Correct,
      Top20_Odds = PGA_Prediction$Top20_Odds,
      Top20_Profit = PGA_Prediction$Top20_Winnings,
      Rating = PGA_Prediction$rating,
      Valid_Odds = PGA_Prediction$Valid_Odds,
      Training_Events_Used = current_event - 1,
      Calibrated_Probability = PGA_Prediction$Calibrated_Probability,
      Kelly_Fraction = PGA_Prediction$Kelly_Fraction,
      Expected_Value = PGA_Prediction$Expected_Value,
      Implied_Probability = PGA_Prediction$Implied_Probability,
      Prob_Edge = PGA_Prediction$Prob_Edge
    )
    
    all_predictions <- rbind(all_predictions, prediction_record)
    
    cat("  Successfully processed event", current_event, "- Players:", nrow(PGA_Prediction), 
        "Betting data:", ifelse(any(PGA_Prediction$Valid_Odds), "Yes", "No"), "\n")
    
  }, error = function(e) {
    cat("  ERROR in prediction for event", current_event, ":", e$message, "\n")
  })
}

cat("\n=== WALK-FORWARD VALIDATION COMPLETE ===\n")
cat("Total predictions:", nrow(all_predictions), "\n")
cat("Events processed:", length(unique(all_predictions$EventNumber)), "\n")

# ===== SAVE WALK-FORWARD RESULTS =====
# Create directory for results
results_dir <- paste0("./Results/", MODEL_NAME, "/")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# Save predictions
write.csv(all_predictions, paste0(results_dir, MODEL_NAME, "_walk_forward_predictions.csv"), row.names = FALSE)

# Calculate and display basic performance metrics
if (nrow(all_predictions) > 0) {
  overall_accuracy <- mean(all_predictions$Prediction_Correct, na.rm = TRUE)
  top20_players <- sum(all_predictions$Actual_Top20, na.rm = TRUE)
  events_with_odds <- length(unique(all_predictions$EventID[all_predictions$Valid_Odds]))
  
  cat("\n=== WALK-FORWARD PERFORMANCE SUMMARY ===\n")
  cat("Overall Accuracy:", round(overall_accuracy * 100, 2), "%\n")
  cat("Total Top20 finishes:", top20_players, "\n")
  cat("Events with betting odds:", events_with_odds, "\n")
  
  # Calibration quality check
  if (ENABLE_CALIBRATION) {
    predictions_with_cal <- all_predictions[!is.na(all_predictions$Calibrated_Probability), ]
    if (nrow(predictions_with_cal) > 0) {
      # Simple calibration check - bin predictions and check actual rates
      predictions_with_cal$prob_bin <- cut(predictions_with_cal$Calibrated_Probability, 
                                           breaks = c(0, 0.05, 0.1, 0.15, 0.2, 0.3, 1), 
                                           include.lowest = TRUE)
      
      calibration_check <- predictions_with_cal %>%
        group_by(prob_bin) %>%
        summarise(
          count = n(),
          avg_predicted_prob = mean(Calibrated_Probability, na.rm = TRUE),
          actual_rate = mean(Actual_Top20, na.rm = TRUE),
          .groups = 'drop'
        ) %>%
        mutate(calibration_error = abs(avg_predicted_prob - actual_rate))
      
      cat("\n=== CALIBRATION CHECK ===\n")
      print(calibration_check)
    }
  }
}

# ===== BETTING SIMULATION =====
if (ENABLE_BETTING_SIMULATION && nrow(all_predictions) > 0) {
  cat("\n=== RUNNING BETTING SIMULATION ===\n")
  
  # Filter for events with valid betting odds
  betting_data <- all_predictions %>%
    filter(Valid_Odds == TRUE, !is.na(Calibrated_Probability), !is.na(Kelly_Fraction))
  
  # Run aggressive betting simulation
  kelly_sim_1 <- simulate_kelly_betting_custom(
    betting_data,
    initial_bankroll = 1000,
    kelly_multiplier = 0.075,
    max_bet_amount = 999,
    max_bet_pct = 1,
    max_event_pct = 1,
    min_kelly_threshold = 0.00,
    min_expected_value = 0.00,
    min_prob_edge = 0.00,
    max_bets_per_event = 999,
    use_dynamic_bankroll = F,
    min_odds = 1,
    max_odds = 999
  )
  
  print_simulation_results(kelly_sim_1)
  
  write.csv(kelly_sim_1$bet_history, paste0(results_dir, MODEL_NAME, "_bet_history.csv"), row.names = FALSE)
  write.csv(kelly_sim_1$bankroll_history, paste0(results_dir, MODEL_NAME, "_bankroll_history.csv"), row.names = FALSE)
  write.csv(kelly_sim_1$parameters, paste0(results_dir, MODEL_NAME, "_parameters.csv"), row.names = FALSE)
  
  # Create comparison plot
  if (require(ggplot2, quietly = TRUE)) {
    tryCatch({
      kelly_sim_1_history <- kelly_sim_1$bankroll_history
      
      # Create plot
      bankroll_plot <- ggplot(kelly_sim_1_history, aes(x = event_number, y = bankroll_end)) +
        geom_line(size = 1) +
        geom_point(alpha = 0.6) +
        geom_hline(yintercept = 1000, linetype = "dashed", alpha = 0.7, color = "gray") +
        labs(
          title = "Simulation Results",
          subtitle = paste("Walk-Forward Validation:", MODEL_NAME),
          x = "Event Number",
          y = "Bankroll (£)"
        ) +
        theme_minimal() +
        theme(
          plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 12)
        )
      
      # Save plot
      print(bankroll_plot)
      
      ggsave(paste0(results_dir, MODEL_NAME, "_bankroll_comparison.png"), 
             bankroll_plot, width = 12, height = 8, dpi = 300)
      
      cat("\nBankroll comparison plot saved to:", paste0(results_dir, MODEL_NAME, "_bankroll_comparison.png"), "\n")
      
    }, error = function(e) {
      cat("Could not create comparison plot:", e$message, "\n")
    })
  }
}
