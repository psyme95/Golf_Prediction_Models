# ===== CONFIGURATION SECTION =====
# Set working directory and parameters
setwd("C:/Projects/Golf")
set.seed(42)
MODEL_NAME <- paste0("Top20_2024_", format(Sys.time(), "%d%m_%H%M"))

# Simulation parameters - moved to top for easy configuration
STARTING_BANKROLL <- 1000
FIXED_STAKE <- 10  # Fixed £10 per bet (changed from percentage)
MIN_PLAYERS_PER_EVENT <- 120

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.7

# ===== HELPER FUNCTIONS =====
# Strategy generation
generate_betting_strategies <- function(size_increments = c(10, 20, 30, 40), log_file = NULL) {
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

  # Log some example strategies for validation
  example_strategies <- head(names(strategies), 5)
  for (strategy_name in example_strategies) {
    strategy <- strategies[[strategy_name]]
  }
  
  return(strategies)
}

# Model training function
# Enhanced model training function with response curves and summary
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
                                             mtry = floor(sqrt(length(available_vars)))
                                             ))
    user.XGBOOST <- list('_allData_allRun' = list(nrounds = 50,
                                                  #eta = 0.1,
                                                  subsample = 0.5,
                                                  colsample_bytree = 0.5,
                                                  min_child_weight = 5,
                                                  print_every_n = 10L
                                                  ))
    user.GAM <- list('_allData_allRun' = list( algo = 'GAM_mgcv',
                                               type = 's_smoother',
                                               k = 6,   #If GAMS not converging, change up to 4, then 5 max
                                               interaction.level = 2,    #Increase to 1 if trouble fitting models 
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
      #OPT.strategy = "bigboss",
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
    
    # ===== SAVE PLOTS AND SUMMARY =====
    # Create directory for plots
    plot_dir <- paste0("./Biomod2_plots/", MODEL_NAME, "/")
    if (!dir.exists(plot_dir)) {
      dir.create(plot_dir, recursive = TRUE)
    }
    
    for(i in myBiomodModelOut@models.computed[grep("allRun",myBiomodModelOut@models.computed)]){
      png(paste0(plot_dir, i, "_response_plots.png"), width=1024, height=1024)
      bm_PlotResponseCurves(myBiomodModelOut, models.chosen = i, do.bivariate=F, fixed.var="mean")
      dev.off()
    }
    
    png(paste0(plot_dir, "ensemble_response_plots.png"), width=1024, height=1024)
    bm_PlotResponseCurves(myBiomodEM)
    dev.off()
    
    # Save model summary
    tryCatch({
      summary_file <- file.path(plot_dir, paste0(MODEL_NAME, "_summary.txt"))
      
      sink(summary_file)
      cat("=== MODEL SUMMARY ===\n")
      cat("Model:", MODEL_NAME, "\n")
      cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
      cat("Variables:", paste(model_vars, collapse = ", "), "\n")
      cat("Training samples:", nrow(train_data), "\n")
      cat("\n=== MODEL PERFORMANCE ===\n")
      print(df.metrics)
      cat("\n=== BUILT MODELS ===\n")
      cat(paste(get_built_models(myBiomodModelOut), collapse = "\n"), "\n")
      sink()
      
      cat("Model summary saved to:", summary_file, "\n")
      
    }, error = function(e) {
      cat("Warning: Could not save summary:", e$message, "\n")
    })
    
    # Save Variable importance
    var.imp <- arrange(get_variables_importance(myBiomodEM), desc(var.imp))
    write.csv(var.imp, paste0("./Evaluation/Variable_Importance_", MODEL_NAME, ".csv"))
    
    return(myBiomodEM)
    
  }, error = function(e) {
    stop(e)
  })
}

# ===== MAIN EXECUTION =====
# Load and validate data (KEEP QUALITY DATA FOR INFORMATION PURPOSES)
eventQuality <- read.csv("./Data/FieldQuality.csv")
eventDates <- read.csv("./Data/EventDates.csv")

df_old <- read_excel("./Data/PGA_processed_features.xlsx")
df_old <- df_old[complete.cases(df_old),]

# Split events into list elements
event_list <- split(df_old, df_old$eventID)
event_list <- event_list[sapply(event_list, function(df) nrow(df) > 80)]

train_event_list <- lapply(event_list, function(df) {
  df %>%
    mutate(Date = dmy(Date)) %>%  # Convert "09-Aug-20" to proper date
    filter(year(Date) <= 2022) %>%
    select(eventID, Date, everything())  # Keep all columns, reorder if needed
})

test_event_list <- lapply(event_list, function(df) {
  df %>%
    mutate(Date = dmy(Date)) %>%  # Convert "09-Aug-20" to proper date
    filter(year(Date) == 2023) %>%
    select(eventID, Date, everything())  # Keep all columns, reorder if needed
})

# Combine all tournament data into one dataframe
df_old <- data.frame()

for (eventID in names(train_event_list)) {
  temp_df <- train_event_list[[eventID]]
  temp_df$eventID <- eventID
  df_old <- rbind(df_old, temp_df)
}

df_new <- data.frame()

for (i in 1:length(test_event_list)) {
  temp_df <- test_event_list[[i]]
  temp_df$eventID <- test_event_list[[i]]$eventID
  df_new <- rbind(df_new, temp_df)
}

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
length(model_vars)

missing_vars <- model_vars[model_vars %not in% names(df_old)]
available_vars_old <- model_vars[model_vars %in% names(df_old)]
available_vars_new <- model_vars[model_vars %in% names(df_new)]
available_vars <- intersect(available_vars_old, available_vars_new)

# cor_matrix <- cor(df_old[, model_vars], use = "complete.obs")
# cor_matrix_numeric <- cor_matrix > 0.75
# cor_matrix_numeric <- ifelse(cor_matrix, 1, 0)
# colSums(cor_matrix_numeric)

# Create datasets (KEEP QUALITY FOR INFORMATION, BUT DON'T USE IN SIMULATION)
train_data <- df_old %>%
  select(playerID, eventID, posn, top_20, all_of(available_vars))

test_data <- df_new %>%
  select(playerID, eventID, posn, top_20, Quality, Top20_Profit, rating, all_of(available_vars))

# Train ensemble model
myBiomodEM <- train_ensemble_model(train_data, available_vars, log_file)
saveRDS(myBiomodEM, file = paste0("./Models/", MODEL_NAME, "_BiomodEM.rds"))
saveRDS(myBiomodEM, file = paste0("./Models/BiomodEM_2024_usethisone.rds"))

# ===== SIMULATION =====
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

# Generate betting strategies
strategies <- generate_betting_strategies(log_file = log_file)

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

# Run event-by-event simulation
for (event_idx in 1:length(new_events)) {
  test_event <- new_events[event_idx]
  
  # Get event data
  event_data <- test_data %>%
    filter(eventID == test_event) %>%
    mutate(Top20_Odds = filter(df_new, eventID == test_event)$Top20_Odds,
           Top20_Profit = filter(df_new, eventID == test_event)$Top20_Profit,
           rating = filter(df_new, eventID == test_event)$rating)
  
  event_info_row <- event_info[event_info$eventID == test_event, ]
  
  if (nrow(event_data) < MIN_PLAYERS_PER_EVENT) {
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
  })
}

# Compile and analyze results
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

# Display final message
cat(sprintf("\n✓ Analysis complete! Results saved to:\n%s\n\n", excel_file))
