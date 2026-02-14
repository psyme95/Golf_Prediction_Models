#### Load Packages ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
'%not in%'  <- function(x,table) is.na(match(x,table,nomatch=NA_integer_))

# Load Data
df <- read.csv("./Data/PGA_withodds.csv")
df$rating <- as.numeric(df$rating)
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df <- df[complete.cases(df),]
eventQuality <- read.csv("./Data/FieldQuality.csv")

df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID")) %>%
  filter(Quality<= 71)

# Create a function to run a single model evaluation and return metrics
evaluate_model <- function(test_event_id) {
  # Set run name for this iteration
  run <- paste0("LOO_", test_event_id)
  
  # Run your existing modeling code here
  pgalist <- split(df, df$eventID)
  
  #### Add more stats for each tournament  #####
  for (eventID in names(pgalist)) {
    pgalist[[eventID]]$rating_normal <- scale(pgalist[[eventID]]$rating)
    pgalist[[eventID]]$rating_mean <- mean(pgalist[[eventID]]$rating, na.rm = TRUE)
    pgalist[[eventID]]$rating_median <- median(pgalist[[eventID]]$rating, na.rm = TRUE)
    pgalist[[eventID]]$diff_from_mean <- pgalist[[eventID]]$rating - mean(pgalist[[eventID]]$rating, na.rm = TRUE)
    pgalist[[eventID]]$diff_from_median <- pgalist[[eventID]]$rating - median(pgalist[[eventID]]$rating, na.rm = TRUE)
    pgalist[[eventID]]$diff_from_max <- pgalist[[eventID]]$rating - max(pgalist[[eventID]]$rating, na.rm = TRUE)
    pgalist[[eventID]]$diff_from_min <- pgalist[[eventID]]$rating - min(pgalist[[eventID]]$rating, na.rm = TRUE)
    pgalist[[eventID]]$top_20 <- ifelse(pgalist[[eventID]]$posn <= 20, 1, 0)
    pgalist[[eventID]]$rating_normal <- scale(pgalist[[eventID]]$rating)
    pgalist[[eventID]]$log_rating <- log(pgalist[[eventID]]$rating+50)
    pgalist[[eventID]]$diff_from_field_quality <-pgalist[[eventID]]$rating - pgalist[[eventID]]$Quality
  }
  
  # Filter to keep only tournaments with enough players
  pgalist_filtered <- pgalist[sapply(pgalist, function(df) nrow(df) > 40)]
  
  # Set aside the test event data
  test_data <- pgalist_filtered[[as.character(test_event_id)]]
  
  # Remove test event from training data
  training_events <- pgalist_filtered[names(pgalist_filtered) != as.character(test_event_id)]
  
  ##### Combine all tournament data into one dataframe for training #####
  all_players_data <- data.frame()
  
  for (eventID in names(training_events)) {
    temp_df <- training_events[[eventID]]
    temp_df$eventID <- eventID
    all_players_data <- rbind(all_players_data, temp_df)
  }
  
  resp <- all_players_data[, "top_20"]
  expl <- all_players_data[, !names(all_players_data) %in% c("eventID", "posn", "playerID","score", "win", "top_20")]
  
  # Setup projection data (the single test event)
  PGA_projection <- test_data
  PGA_projection$eventID <- as.character(test_event_id)
  resp.proj <- PGA_projection[, c("playerID","eventID","top_20", "Top20_Odds", "Top40_Odds")]
  expl.proj <- PGA_projection[, !names(all_players_data) %in% c("eventID", "posn", "playerID","score", "win", "top_20")]
  
  #### Variable Selection ####
  keepLayers <- names(expl)[c(66,75,7,8,21,25,26,30,33,49,61,64)] # Run 2.EQ70.Top20 - Bigboss model settings - Field quality added - Larger split than run 1
  
  # drop the layers
  myExpl_prediction <- expl[keepLayers]
  names(myExpl_prediction)

  reps  <-  1 # CHANGE NO OF MODEL REPS. This should BE 5 fold for exploratory models, 1 fold for the final run.
  
  models.to.proc = c('GAM','GLM','RF','FDA','CTA','ANN','GBM','MAXNET','XGBOOST') 
  
  ##### Convert Categorical Variables to Factors ####
  # If using categorical variables you cannot use MARS or SRE as modelling options
  factors.to.convert <- which(names(myExpl_prediction)=="Starts_Not10",
                              names(myExpl_prediction)=="playerID")
  
  for(j in factors.to.convert){
    myExpl_prediction[[j]] <- as.factor(myExpl_prediction[[j]])
  }
  
  # Create Biomod Data Object
  myBiomodData  <-  BIOMOD_FormatingData(
    resp.var = resp,
    expl.var = myExpl_prediction,
    #eval.resp.var = resp.test,
    #eval.expl.var = expl.test,
    resp.name = "PGA")
  
  # Add dummy coordinates (sequential numbers)
  dummy_coords <- data.frame(
    x = 1:length(myBiomodData@data.species),
    y = 1:length(myBiomodData@data.species)
  )
  
  # Update biomod data object with these coordinates
  myBiomodData@coord <- dummy_coords
  
  #### Modelling ####
  # Individual Models
  myBiomodModelOut  <-  BIOMOD_Modeling(
    bm.format=myBiomodData,
    models = models.to.proc,
    OPT.strategy = "bigboss",
    CV.nb.rep=reps, 
    CV.perc=0.75,   
    weights=NULL,
    var.import=0,
    metric.eval = c('TSS'),
    nb.cpu = 1,
    do.progress = F
  )
  
  ##### Evaluate the Single Models #####
  myBiomodModelEval  <-  get_evaluations(myBiomodModelOut)

  #create TSS scores table
  allRun.model.names <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"full.name"]
  allRun.model.sens <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"sensitivity"]
  allRun.model.spec <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"specificity"]
  allRun.model.TSS <- (allRun.model.sens/100+allRun.model.spec/100)-1
  
  df.metrics  <-  data.frame(matrix(0, ncol = length(allRun.model.TSS), nrow = 1))
  names(df.metrics) <- allRun.model.names
  row.names(df.metrics) <- "TSS"
  df.metrics[1,1:length(allRun.model.TSS)] <- allRun.model.TSS
  
  #### Create the ENSEMBLE MODEL ####
  gc()
  
  myBiomodEM  <-  BIOMOD_EnsembleModeling(
    bm.mod = myBiomodModelOut,
    models.chosen = allRun.model.names, #Include all FULL MODELS
    em.algo = c('EMwmean'), #'EMci','EMcv'), #('EMmean', 'EMcv', 'EMci', 'EMmedian', 'EMca' or 'EMwmean') ('prob.median', 'prob.cv', 'prob.ci', 'committee.averaging', 'prob.mean.weight')
    metric.select = c("user.defined"),
    metric.select.thresh = c(0.2), #lowered to 0.5 because 0.6 is difficult to achieve.  If ensemmble is poor though, change to 0.6
    metric.select.table = df.metrics,
    metric.eval = c('TSS', 'ROC'),
    var.import = 3,
    EMwmean.decay = 'proportional',
  )

  # ####  Project the ensemble model to testing tournament as check how good it is ####
  resp.proj.xy <- resp.proj[,1:2]
  
  myExpl_projection <- expl.proj[keepLayers]
  names(myExpl_projection)
  
  # Ensemble forecasting
  myBiomodProj <- BIOMOD_EnsembleForecasting(
    bm.em = myBiomodEM,
    bm.proj = NULL,
    proj.name = "wmean",
    new.env = myExpl_projection,
    new.env.xy = resp.proj.xy,
    models.chosen = "all",
    metric.binary = "TSS",
    metric.filter = "TSS",
    na.rm = TRUE
  )
  
  # Format prediction data
  PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
  colnames(PGA_Prediction)[3] <- "Model_Score"
  
  # Define thresholds we want to test
  thresholds <- c(5,10,15,20)
  stake_types <- c("fixed", "flex")
  fixed_stake_amount <- 10
  target_profit <- 10
  
  #### Process test event ####
  # Join prediction data with actual results
  PGA_Prediction <- merge(
    PGA_Prediction,
    PGA_projection[, c("eventID", "playerID", "Top20_Odds", "posn")],
    by = c("eventID", "playerID"),
    all.x = TRUE
  )
  
  # Add actual Top20 finish column
  PGA_Prediction$Top20 <- ifelse(PGA_Prediction$posn <= 20, 1, 0)
  
  # Setup odds validity
  PGA_Prediction$Valid_Odds <- PGA_Prediction$Top20_Odds > 1
  PGA_Prediction$Fixed_Stake <- fixed_stake_amount
  PGA_Prediction$Flex_Stake <- ifelse(
    PGA_Prediction$Valid_Odds,
    round(target_profit / (PGA_Prediction$Top20_Odds - 1), 2),
    0
  )
  
  # Calculate thresholds and create model score columns
  for (t in thresholds) {
    # Determine threshold value
    threshold_value <- sort(PGA_Prediction$Model_Score, decreasing = TRUE)[min(t, length(PGA_Prediction$Model_Score))]
    
    # Create model score column
    col_name <- paste0("Top", t, "_ModelScore")
    PGA_Prediction[[col_name]] <- ifelse(PGA_Prediction$Model_Score >= threshold_value, 1, 0)
    
    # Calculate if model was correct
    correct_col <- paste0("Top", t, "_ModelScore_Correct")
    PGA_Prediction[[correct_col]] <- ifelse(PGA_Prediction$Top20 == PGA_Prediction[[col_name]], 1, 0)
    
    # Process each stake type
    for (stake_type in stake_types) {
      stake_col_suffix <- ifelse(stake_type == "fixed", "", "_Flex")
      stake_amount_col <- ifelse(stake_type == "fixed", "Fixed_Stake", "Flex_Stake")
      
      # Calculate stake placed
      stake_placed_col <- paste0("Top", t, "_Stake_Placed", stake_col_suffix)
      PGA_Prediction[[stake_placed_col]] <- ifelse(
        PGA_Prediction[[col_name]] == 1 & PGA_Prediction$Valid_Odds,
        PGA_Prediction[[stake_amount_col]],
        0
      )
      
      # Calculate winnings
      winnings_col <- paste0("Top", t, "_Player_Winnings", stake_col_suffix)
      PGA_Prediction[[winnings_col]] <- ifelse(
        PGA_Prediction[[col_name]] == 1 & 
          PGA_Prediction$Valid_Odds & 
          PGA_Prediction[[correct_col]] == 1,
        PGA_Prediction[[stake_amount_col]] * PGA_Prediction$Top20_Odds,
        0
      )
      
      # Calculate profit
      profit_col <- paste0("Top", t, "_Player_Profit", stake_col_suffix)
      PGA_Prediction[[profit_col]] <- PGA_Prediction[[winnings_col]] - PGA_Prediction[[stake_placed_col]]
    }
  }
  
  # Function to create column name for summary metrics
  col_name <- function(metric, threshold, stake_type = NULL) {
    if (is.null(stake_type)) {
      return(paste0(metric, "_top", threshold))
    } else {
      return(paste0(metric, "_", stake_type, "_top", threshold))
    }
  }
  
  # Calculate event summary
  event_id <- unique(PGA_Prediction$eventID)[1]
  valid_data <- PGA_Prediction[PGA_Prediction$Valid_Odds, ]
  
  # Initialize event summary with event ID
  event_summary <- data.frame(eventID = event_id)
  
  # Calculate summaries for each threshold
  for (t in thresholds) {
    # Base metrics
    model_score_col <- paste0("Top", t, "_ModelScore")
    correct_col <- paste0("Top", t, "_ModelScore_Correct")
    
    # Calculate betting totals and success rates
    total_bets <- sum(valid_data[[model_score_col]] == 1)
    successful_bets <- sum(valid_data[[model_score_col]] == 1 & valid_data[[correct_col]] == 1)
    success_rate <- ifelse(total_bets > 0, round(successful_bets / total_bets * 100, 2), 0)
    
    # Add to summary
    event_summary[[col_name("bets", t)]] <- total_bets
    event_summary[[col_name("successful", t)]] <- successful_bets
    event_summary[[col_name("success_rate", t)]] <- success_rate
    
    # Calculate for each stake type
    for (stake_type in stake_types) {
      stake_suffix <- ifelse(stake_type == "fixed", "", "_Flex")
      
      # Get column names
      stake_col <- paste0("Top", t, "_Stake_Placed", stake_suffix)
      winnings_col <- paste0("Top", t, "_Player_Winnings", stake_suffix)
      profit_col <- paste0("Top", t, "_Player_Profit", stake_suffix)
      
      # Calculate totals
      total_stake <- sum(valid_data[[stake_col]], na.rm = TRUE)
      total_winnings <- sum(valid_data[[winnings_col]], na.rm = TRUE)
      total_profit <- sum(valid_data[[profit_col]], na.rm = TRUE)
      roi <- ifelse(total_stake > 0, round(total_profit / total_stake * 100, 2), 0)
      
      # Add to summary
      event_summary[[col_name("stake", t, stake_type)]] <- round(total_stake, 2)
      event_summary[[col_name("winnings", t, stake_type)]] <- round(total_winnings, 2)
      event_summary[[col_name("profit", t, stake_type)]] <- round(total_profit, 2)
      event_summary[[col_name("roi", t, stake_type)]] <- roi
    }
  }
  
  # Calculate run metrics for this event
  run_metrics <- data.frame()
  
  # Calculate metrics for each threshold
  for (t in thresholds) {
    # Column names for this threshold
    success_rate_col <- paste0("success_rate_top", t)
    fixed_roi_col <- paste0("roi_fixed_top", t)
    flex_roi_col <- paste0("roi_flex_top", t)
    bets_col <- paste0("bets_top", t)
    fixed_profit_col <- paste0("profit_fixed_top", t)
    flex_profit_col <- paste0("profit_flex_top", t)
    
    # Check if we have bets for this threshold
    if (event_summary[[bets_col]] > 0) {
      # Calculate statistics
      threshold_metrics <- data.frame(
        EventID = event_id,
        Run = run,
        Threshold = paste0("Top", t),
        Success_Rate = event_summary[[success_rate_col]],
        Fixed_ROI = event_summary[[fixed_roi_col]],
        Flex_ROI = event_summary[[flex_roi_col]],
        Fixed_Profit = event_summary[[fixed_profit_col]],
        Flex_Profit = event_summary[[flex_profit_col]],
        Profitable_Fixed = event_summary[[fixed_profit_col]] > 0,
        Profitable_Flex = event_summary[[flex_profit_col]] > 0,
        Num_Bets = event_summary[[bets_col]]
      )
      
      # Add to run metrics
      run_metrics <- rbind(run_metrics, threshold_metrics)
    }
  }
  
  return(run_metrics)
}

# Create empty dataframe to store all results
all_model_metrics <- data.frame()

# Get all filtered event IDs
pgalist <- split(df, df$eventID)
pgalist_filtered <- pgalist[sapply(pgalist, function(df) nrow(df) > 40)]
all_event_ids <- names(pgalist_filtered)[1:3]

# Run leave-one-out cross-validation
for (test_event_id in all_event_ids) {
  cat("Evaluating model on event", test_event_id, "\n")
  
  # Run model and get metrics
  run_metrics <- evaluate_model(test_event_id)
  
  # Add to overall results
  all_model_metrics <- rbind(all_model_metrics, run_metrics)
  
  # Save intermediate results after each event
  write.csv(all_model_metrics, "model_evaluation_metrics_loo.csv", row.names = FALSE)
  
  cat("Completed model evaluation on event", test_event_id, "\n")
}

# Calculate aggregate statistics across all events
aggregate_metrics <- all_model_metrics %>%
  group_by(Threshold) %>%
  summarize(
    Avg_Success_Rate = mean(Success_Rate, na.rm = TRUE),
    Avg_Fixed_ROI = mean(Fixed_ROI, na.rm = TRUE),
    Avg_Flex_ROI = mean(Flex_ROI, na.rm = TRUE),
    Total_Fixed_Profit = sum(Fixed_Profit, na.rm = TRUE),
    Total_Flex_Profit = sum(Flex_Profit, na.rm = TRUE),
    Profitable_Events_Fixed = sum(Profitable_Fixed, na.rm = TRUE),
    Profitable_Events_Flex = sum(Profitable_Flex, na.rm = TRUE),
    Total_Events = n()
  )

# Calculate percentages of profitable events
aggregate_metrics$Pct_Events_Profitable_Fixed <- round(aggregate_metrics$Profitable_Events_Fixed / aggregate_metrics$Total_Events * 100, 2)
aggregate_metrics$Pct_Events_Profitable_Flex <- round(aggregate_metrics$Profitable_Events_Flex / aggregate_metrics$Total_Events * 100, 2)

# Print the aggregate metrics
print(aggregate_metrics)

# Save final results
write.csv(all_model_metrics, "A:/OneDrive - University of Southampton/Golf/Betting Simulations/Top 20/PGA_Top20_Under71EQ_LOOCV_AllEvents.csv", row.names = FALSE)
write.csv(aggregate_metrics, "A:/OneDrive - University of Southampton/Golf/Betting Simulations/Top 20/PGA_Top20_Under71EQ_LOOCV_AggregatedEvents.csv", row.names = FALSE)
