# ===== CONFIGURATION SECTION =====
# Set working directory and parameters
setwd("C:/Projects/Golf")
set.seed(42)

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.7
MODEL_NAME <- paste0("Top20_RW_", format(Sys.time(), "%d%m_%H%M"))
MIN_PLAYERS_PER_EVENT <- 1

# Rolling window parameters
ROLLING_WINDOW_SIZE <- 80  # Number of events to include in training window
RETRAIN_FREQUENCY <- 1     # Retrain after every N events
MIN_PREDICTION_EVENT <- ROLLING_WINDOW_SIZE + 1 # First event to start making predictions (window size + 1)

# ===== HELPER FUNCTIONS =====
# Model training function (unchanged)
train_ensemble_model <- function(train_data, model_vars, log_file = NULL) {
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
    
  }

# Function to get rolling window training data
get_rolling_window_data <- function(df_all, current_event, window_size, available_vars) {
  # Calculate window boundaries
  window_start <- max(1, current_event - window_size)
  window_end <- current_event - 1
  
  cat("Rolling window: events", window_start, "to", window_end, 
      "(", window_end - window_start + 1, "events)\n")
  
  # Extract training data from the rolling window
  train_data <- df_all %>%
    filter(event_number >= window_start & event_number <= window_end) %>%
    select(playerID, eventID, posn, top_20, Top20_odds, all_of(available_vars))
  
  return(train_data)
}

# Kelly betting simulation function (unchanged from original)
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
      if (is.na(row$Calibrated_Probability) || is.na(row$Top20_odds) || 
          row$Top20_odds <= 0 || !row$Valid_odds) {
        next
      }
      
      # Calculate probability edge over market
      implied_prob <- 1 / row$Top20_odds
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
          row$Top20_odds >= min_odds &&
          row$Top20_odds <= max_odds
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
          odds = row$Top20_odds,
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
        potential_bets$final_bet_amount <- round(potential_bets$constrained_bet * scaling_factor, 0)
        potential_bets$event_scaled <- TRUE
      } else {
        potential_bets$final_bet_amount <- round(potential_bets$constrained_bet, 0)
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

# Function to print simulation results (unchanged)
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
    cat("  Average odds:", round(sim_result$avg_odds, 2), "\n")
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
df_all <- read_excel("./Data/PGA_070725_Processed.xlsx")
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

# Define model variables
model_vars <- c("rating_vs_field_best",
                "rating",
                "rating_vs_field_worst",
                "yr3_All",
                "Top40_odds",
                "Top20_odds",
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

# Check available variables
missing_vars <- model_vars[model_vars %not in% names(df_all)]
available_vars <- model_vars[model_vars %in% names(df_all)]

cat("Available variables:", length(available_vars), "/", length(model_vars), "\n")
if (length(missing_vars) > 0) {
  cat("Missing variables:", paste(missing_vars, collapse = ", "), "\n")
}

# ===== ROLLING WINDOW VALIDATION WITH CALIBRATION =====
# Initialize tracking
all_predictions <- data.frame()
trained_models <- list()
trained_calibration_models <- list()
last_training_event <- 0

# Rolling window loop - start from MIN_PREDICTION_EVENT
for (current_event in MIN_PREDICTION_EVENT:total_events) {
  
  current_event_id <- unique_events$eventID[unique_events$event_number == current_event]
  cat("Processing event", current_event, "of", total_events, ":", current_event_id, "\n")
  
  # Check if we need to retrain
  need_retrain <- (current_event - last_training_event) >= RETRAIN_FREQUENCY
  
  if (need_retrain || length(trained_models) == 0) {
    cat("Retraining model and calibration...\n")
    
    # Get rolling window training data
    train_data <- get_rolling_window_data(df_all, current_event, ROLLING_WINDOW_SIZE, available_vars)
    
    # Train ensemble model
    current_model <- train_ensemble_model(train_data, available_vars)
    trained_models[[as.character(current_event)]] <- current_model
    
    # Train calibration model using rolling window
    calibration_model <- NULL
    
    # Get predictions from the ensemble model on training data
    logit_model_data <- data.frame(
      EventID = train_data$eventID,
      PlayerID = train_data$playerID,
      Model_Score = current_model@models.prediction@val$pred,
      Actual_Top20 = train_data$top_20,
      Top20_odds = 1 / train_data$Top20_odds
    )
        
     # Train calibration model
     calibration_model <- glm(Actual_Top20 ~ Model_Score + Top20_odds,
                              data = logit_model_data,
                              family = binomial())
      
     # calibration_model <- glm(Actual_Top20 ~ Model_Score,
     #                          data = logit_model_data,
     #                          family = binomial())
  }
  
  last_training_event <- current_event

  
  # Test data: current event only
  test_columns <- c("playerID", "eventID", "posn", "top_20", "Quality", 
                    "Top20_odds", "Top20_Profit", "rating", available_vars)
  test_columns <- unique(test_columns)
  available_test_columns <- test_columns[test_columns %in% names(df_all)]
  
  test_data <- df_all %>%
    filter(event_number == current_event) %>%
    select(all_of(available_test_columns))
  
  # Skip if insufficient players
  if (nrow(test_data) < MIN_PLAYERS_PER_EVENT) {
    cat("  Skipping - insufficient players:", nrow(test_data), "\n")
    next
  }
  
  # Make predictions for this event
  cat("  Making predictions for", nrow(test_data), "players...\n")

  test_expl <- test_data[, available_vars]
  test_resp.xy <- test_data[, c("playerID", "eventID")]

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
  test_prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
  colnames(test_prediction)[3] <- "Model_Score"
    
  # Merge with test data
  essential_cols <- c("eventID", "playerID", "posn", "top_20")
  merge_cols <- intersect(names(test_data), c(essential_cols, "Top20_odds", "Top20_Profit", "rating", "Quality"))
  
  test_prediction <- merge(
    test_prediction,
    test_data[, merge_cols],
    by = c("eventID", "playerID"), all.x = TRUE
  )
  
  # Apply calibration
  cat("  Applying calibration...\n")
    
  calibration_data <- data.frame(
    Model_Score = test_prediction$Model_Score,
    Top20_odds = test_prediction$Top20_odds
  )

  test_prediction$Calibrated_Probability <- predict(calibration_model, 
                                                     calibration_data, 
                                                     type = "response")

  # Calculate derived metrics
  test_prediction$Valid_odds <- test_prediction$Top20_odds > 1
  test_prediction$Model_Rank <- rank(-test_prediction$Model_Score, ties.method = "first")
  test_prediction$Prediction_Correct <- ifelse(
    (test_prediction$top_20 == 1 & test_prediction$Model_Rank <= 20) | 
      (test_prediction$top_20 == 0 & test_prediction$Model_Rank > 20), 
    1, 0)
    
  # Add Kelly calculations
  test_prediction <- test_prediction %>%
    mutate(
      # odds are already in decimal format
      decimal_odds = Top20_odds,
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
  
  # Store results with rolling window metadata
  prediction_record <- data.frame(
    EventNumber = current_event,
    EventID = test_prediction$eventID,
    PlayerID = test_prediction$playerID,
    Model_Score = round(test_prediction$Model_Score, 4),
    Model_Rank = test_prediction$Model_Rank,
    Actual_Top20 = test_prediction$top_20,
    Actual_Position = test_prediction$posn,
    Prediction_Correct = test_prediction$Prediction_Correct,
    Top20_odds = test_prediction$Top20_odds,
    Top20_Profit = test_prediction$Top20_Profit,
    Rating = test_prediction$rating,
    Valid_odds = test_prediction$Valid_odds,
    Training_Window_Start = max(1, current_event - ROLLING_WINDOW_SIZE),
    Training_Window_End = current_event - 1,
    Training_Events_Used = min(ROLLING_WINDOW_SIZE, current_event - 1),
    Calibrated_Probability = test_prediction$Calibrated_Probability,
    Kelly_Fraction = test_prediction$Kelly_Fraction,
    Expected_Value = test_prediction$Expected_Value,
    Implied_Probability = test_prediction$Implied_Probability,
    Prob_Edge = test_prediction$Prob_Edge
  )
    
  all_predictions <- rbind(all_predictions, prediction_record)
    
  cat("Successfully processed event", current_event, "\n",
      "Field Size:", nrow(test_prediction), "\n", 
      "Training Window:", prediction_record$Training_Window_Start[1], "to", prediction_record$Training_Window_End[1], "\n")
  }

# ===== SAVE ROLLING WINDOW RESULTS =====
# Create directory for results
results_dir <- paste0("./Results/", MODEL_NAME, "/")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# Save predictions
write.csv(all_predictions, paste0(results_dir, "Model_Predictions.csv"), row.names = FALSE)

# ===== BETTING SIMULATION =====
cat("\n=== RUNNING BETTING SIMULATION ===\n")
  
# Filter for events with valid betting odds
betting_data <- all_predictions %>%
  filter(Valid_odds == TRUE, !is.na(Calibrated_Probability), !is.na(Kelly_Fraction))

cat("Betting data available for", nrow(betting_data), "players across", 
    length(unique(betting_data$EventNumber)), "events\n")
  
# Run rolling window betting simulation
kelly_sim_rolling <- simulate_kelly_betting_custom(
  betting_data,
  initial_bankroll = 1000,
  kelly_multiplier = 0.075,
  max_bet_amount = 999,
  max_bet_pct = 1,
  max_event_pct = 1,
  min_kelly_threshold = 0.00,
  min_expected_value = 0.00,
  min_prob_edge = 0.00,
  max_bets_per_event = 5,
  use_dynamic_bankroll = FALSE,
  min_odds = 1,
  max_odds = 999
)
  
print_simulation_results(kelly_sim_rolling)
  
# Save simulation results
write.csv(kelly_sim_rolling$bet_history, paste0(results_dir, "Bet_History.csv"), row.names = FALSE)
write.csv(kelly_sim_rolling$bankroll_history, paste0(results_dir, "Bankroll_History.csv"), row.names = FALSE)

# Save parameters as a proper data frame
params_df <- data.frame(
  Parameter = names(kelly_sim_rolling$parameters),
  Value = unlist(kelly_sim_rolling$parameters),
  stringsAsFactors = FALSE
)

write.csv(params_df, paste0(results_dir, "Parameters.csv"), row.names = FALSE)

# Create comprehensive plot
bankroll_plot <- ggplot(kelly_sim_rolling$bankroll_history, aes(x = event_number, y = bankroll_end)) +
  geom_line(size = 1, color = "steelblue") +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_hline(yintercept = 1000, linetype = "dashed", alpha = 0.7, color = "gray") +
  labs(
    title = "Rolling Window Betting Simulation Results",
    subtitle = paste("Model:", MODEL_NAME, "| Window Size:", ROLLING_WINDOW_SIZE, "events"),
    x = "Event Number",
    y = "Bankroll (£)",
    caption = paste("Final ROI:", round(kelly_sim_rolling$roi, 2), "% | Total Bets:", kelly_sim_rolling$total_bets)
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    plot.caption = element_text(size = 10, color = "gray50")
  )
      
# Save plot
print(bankroll_plot)

ggsave(paste0(results_dir, MODEL_NAME, "_rolling_window_performance.png"), 
       bankroll_plot, width = 12, height = 8, dpi = 300)
  
cat("\nRolling window performance plot saved to:", paste0(results_dir, MODEL_NAME, "_rolling_window_performance.png"), "\n")
