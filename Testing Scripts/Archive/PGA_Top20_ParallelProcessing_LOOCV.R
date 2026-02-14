#### Load Packages ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
library(foreach)
library(doParallel)

'%not in%'  <- function(x,table) is.na(match(x,table,nomatch=NA_integer_))

# Define global constants
fixed_stake_amount <- 10
target_profit <- 10

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
  # Start with garbage collection
  gc(reset = TRUE)
  
  # Set run name for this iteration
  run <- paste0("LOO_", test_event_id)
  
  # Add debug logging
  log_message <- function(msg) {
    cat(paste0(Sys.time(), " - Event ", test_event_id, ": ", msg, "\n"), 
        file = paste0("A:/OneDrive - University of Southampton/Golf/temp_status/event_", test_event_id, ".log"), append = TRUE)
  }
  
  log_message("Starting evaluation")
  
  # Run your existing modeling code with additional checks
  pgalist <- split(df, df$eventID)
  log_message(paste("Split data into", length(pgalist), "events"))
  
  # Check if test_event_id exists in pgalist
  if (!test_event_id %in% names(pgalist)) {
    log_message(paste("ERROR: Event ID", test_event_id, "not found in pgalist"))
    return(NULL)
  }
  
  #### Add more stats for each tournament with robust checks #####
  for (eventID in names(pgalist)) {
    tryCatch({
      # Check for empty data frames
      if (nrow(pgalist[[eventID]]) == 0) {
        log_message(paste("WARNING: Event", eventID, "has 0 rows"))
        next
      }
      
      # Check if rating column exists and has values
      if (!"rating" %in% names(pgalist[[eventID]]) || all(is.na(pgalist[[eventID]]$rating))) {
        log_message(paste("WARNING: Event", eventID, "missing rating data"))
        next
      }
      
      # Continue with your calculations with NA handling
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
      
      # Only calculate if Quality column exists
      if ("Quality" %in% names(pgalist[[eventID]])) {
        pgalist[[eventID]]$diff_from_field_quality <- pgalist[[eventID]]$rating - pgalist[[eventID]]$Quality
      } else {
        log_message(paste("WARNING: Event", eventID, "missing Quality column"))
      }
    }, error = function(e) {
      log_message(paste("ERROR processing event", eventID, ":", conditionMessage(e)))
    })
  }
  
  log_message("Filtering to keep only tournaments with enough players")
  # Filter to keep only tournaments with enough players
  pgalist_filtered <- pgalist[sapply(pgalist, function(df) nrow(df) > 40)]
  log_message(paste("Filtered to", length(pgalist_filtered), "events"))
  
  # Check if test_event_id exists in filtered list
  if (!test_event_id %in% names(pgalist_filtered)) {
    log_message(paste("ERROR: Event ID", test_event_id, "not found in filtered pgalist"))
    return(NULL)
  }
  
  # Set aside the test event data
  test_data <- pgalist_filtered[[as.character(test_event_id)]]
  log_message(paste("Test data contains", nrow(test_data), "rows"))
  
  # Remove test event from training data
  training_events <- pgalist_filtered[names(pgalist_filtered) != as.character(test_event_id)]
  log_message(paste("Training data contains", length(training_events), "events"))
  
  ##### Combine all tournament data into one dataframe for training #####
  all_players_data <- data.frame()
  
  for (eventID in names(training_events)) {
    tryCatch({
      temp_df <- training_events[[eventID]]
      temp_df$eventID <- eventID
      all_players_data <- rbind(all_players_data, temp_df)
    }, error = function(e) {
      log_message(paste("ERROR combining data for event", eventID, ":", conditionMessage(e)))
    })
  }
  
  log_message(paste("Combined training data contains", nrow(all_players_data), "rows"))
  
  # Check if we have enough combined data
  if (nrow(all_players_data) < 100) {
    log_message("ERROR: Not enough combined training data")
    return(NULL)
  }
  
  # Verify all required columns exist
  required_cols <- c("top_20", "eventID", "posn", "playerID", "score", "win")
  for (col in required_cols) {
    if (!col %in% names(all_players_data)) {
      log_message(paste("ERROR: Missing required column", col, "in training data"))
      return(NULL)
    }
  }
  
  resp <- all_players_data[, "top_20"]
  expl <- all_players_data[, !names(all_players_data) %in% c("eventID", "posn", "playerID", "score", "win", "top_20")]
  
  log_message(paste("Response variable has", length(resp), "entries"))
  log_message(paste("Explanatory variables have", ncol(expl), "columns"))
  
  # Setup projection data (the single test event)
  PGA_projection <- test_data
  PGA_projection$eventID <- as.character(test_event_id)
  
  # Verify all required columns exist in test data
  for (col in c("playerID", "eventID", "top_20", "Top20_Odds", "Top40_Odds")) {
    if (!col %in% names(PGA_projection)) {
      log_message(paste("ERROR: Missing required column", col, "in test data"))
      return(NULL)
    }
  }
  
  resp.proj <- PGA_projection[, c("playerID", "eventID", "top_20", "Top20_Odds", "Top40_Odds")]
  expl.proj <- PGA_projection[, !names(all_players_data) %in% c("eventID", "posn", "playerID", "score", "win", "top_20")]
  
  #### Variable Selection ####
  keepLayers <- names(expl)[c(66,75,7,8,21,25,26,30,33,49,61,64)] # Run 2.EQ70.Top20 - Bigboss model settings
  keepLayers <- names(expl)[c(66,75,7,8,15,21,25,26,30,33,49,51,61,63,64,69,73)] # Run 5.EQ70.Top20 - Bigboss model settings - Field quality added - Larger split than run 1
  keepLayers <- c("Quality", "diff_from_field_quality", "current_top5", "current_top20", "course_top20", "location_top20", "compat2_rank", "Starts",
                  "AvPosn_Rank","X_Top10", "sgt2g_rank", "sgt2g_top20", "sgp_rank", "sgp_top20", "Top20_Odds", "rating_median", "diff_from_min") # Run 6.EQ70.Top20 - Bigboss model settings - Field quality added - Larger split than run 1
  
  log_message(paste("Using", length(keepLayers), "variables for modeling"))
  
  # Check if all keepLayers exist
  missing_layers <- keepLayers[keepLayers %not in% names(expl)]
  if (length(missing_layers) > 0) {
    log_message(paste("ERROR: Missing required layers in explanatory variables:", paste(missing_layers, collapse=", ")))
    return(NULL)
  }
  
  # drop the layers
  myExpl_prediction <- expl[keepLayers]
  log_message(paste("Selected explanatory variables have", ncol(myExpl_prediction), "columns"))
  
  reps <- 1 # CHANGE NO OF MODEL REPS. This should BE 5 fold for exploratory models, 1 fold for the final run.
  
  models.to.proc = c('GAM', 'GLM', 'RF', 'FDA', 'CTA', 'ANN', 'GBM', 'MAXNET', 'XGBOOST') 
  
  ##### Convert Categorical Variables to Factors ####
  # If using categorical variables you cannot use MARS or SRE as modelling options
  factors.to.convert <- which(names(myExpl_prediction)=="Starts_Not10", 
                              names(myExpl_prediction)=="playerID")
  
  for(j in factors.to.convert){
    if (j <= ncol(myExpl_prediction)) {
      myExpl_prediction[[j]] <- as.factor(myExpl_prediction[[j]])
    }
  }
  
  log_message("Creating Biomod Data Object")
  # Create Biomod Data Object
  tryCatch({
    myBiomodData <- BIOMOD_FormatingData(
      resp.var = resp,
      expl.var = myExpl_prediction,
      resp.name = "PGA")
    
    # Add dummy coordinates (sequential numbers)
    dummy_coords <- data.frame(
      x = 1:length(myBiomodData@data.species),
      y = 1:length(myBiomodData@data.species)
    )
    
    # Update biomod data object with these coordinates
    myBiomodData@coord <- dummy_coords
    
    log_message("Data formatting complete")
  }, error = function(e) {
    log_message(paste("ERROR in BIOMOD_FormatingData:", conditionMessage(e)))
    return(NULL)
  })
  
  #### Modelling ####
  log_message("Starting modeling")
  # Individual Models
  tryCatch({
    myBiomodModelOut <- BIOMOD_Modeling(
      bm.format = myBiomodData,
      models = models.to.proc,
      OPT.strategy = "bigboss",
      CV.nb.rep = reps, 
      CV.perc = 0.75,   
      weights = NULL,
      var.import = 0,
      metric.eval = c('TSS'),
      modeling.id = paste0("model_", test_event_id),
      nb.cpu = 1,  # This is kept at 1 because we're parallelizing at a higher level
      do.progress = F
    )
    log_message("BIOMOD_Modeling complete")
  }, error = function(e) {
    log_message(paste("ERROR in BIOMOD_Modeling:", conditionMessage(e)))
    return(NULL)
  })
  
  ##### Evaluate the Single Models #####
  log_message("Evaluating models")
  tryCatch({
    myBiomodModelEval <- get_evaluations(myBiomodModelOut)
    
    #create TSS scores table
    allRun.model.names <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun" & 
                                                    myBiomodModelEval$metric.eval=="TSS"), "full.name"]
    allRun.model.sens <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun" & 
                                                   myBiomodModelEval$metric.eval=="TSS"), "sensitivity"]
    allRun.model.spec <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun" & 
                                                   myBiomodModelEval$metric.eval=="TSS"), "specificity"]
    allRun.model.TSS <- (allRun.model.sens/100 + allRun.model.spec/100) - 1
    
    df.metrics <- data.frame(matrix(0, ncol = length(allRun.model.TSS), nrow = 1))
    names(df.metrics) <- allRun.model.names
    row.names(df.metrics) <- "TSS"
    df.metrics[1,1:length(allRun.model.TSS)] <- allRun.model.TSS
    
    log_message("Model evaluation complete")
  }, error = function(e) {
    log_message(paste("ERROR in model evaluation:", conditionMessage(e)))
    return(NULL)
  })
  
  #### Create the ENSEMBLE MODEL ####
  log_message("Creating ensemble model")
  gc()
  
  tryCatch({
    myBiomodEM <- BIOMOD_EnsembleModeling(
      bm.mod = myBiomodModelOut,
      models.chosen = allRun.model.names, #Include all FULL MODELS
      em.algo = c('EMwmean'), #'EMci','EMcv'), 
      metric.select = c("user.defined"),
      metric.select.thresh = c(0.2), 
      metric.select.table = df.metrics,
      metric.eval = c('TSS', 'ROC'),
      var.import = 3,
      EMwmean.decay = 'proportional',
    )
    
    log_message("Ensemble modeling complete")
  }, error = function(e) {
    log_message(paste("ERROR in BIOMOD_EnsembleModeling:", conditionMessage(e)))
    return(NULL)
  })
  
  # ####  Project the ensemble model to testing tournament as check how good it is ####
  log_message("Projecting ensemble model")
  resp.proj.xy <- resp.proj[,1:2]
  
  # Check if all required keepLayers exist in expl.proj
  missing_proj_layers <- keepLayers[keepLayers %not in% names(expl.proj)]
  if (length(missing_proj_layers) > 0) {
    log_message(paste("ERROR: Missing required layers in projection variables:", 
                      paste(missing_proj_layers, collapse=", ")))
    return(NULL)
  }
  
  myExpl_projection <- expl.proj[keepLayers]
  log_message(paste("Projection data has", nrow(myExpl_projection), "rows and", 
                    ncol(myExpl_projection), "columns"))
  
  # Ensemble forecasting
  tryCatch({
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
    
    log_message("Ensemble forecasting complete")
  }, error = function(e) {
    log_message(paste("ERROR in BIOMOD_EnsembleForecasting:", conditionMessage(e)))
    return(NULL)
  })
  
  # Format prediction data
  log_message("Formatting prediction results")
  tryCatch({
    PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
    colnames(PGA_Prediction)[3] <- "Model_Score"
    
    # Define thresholds we want to test
    thresholds <- c(5,10,15,20,25,30,35,40)
    stake_types <- c("fixed", "flex")
    
    log_message("Joining prediction with actual results")
    # Join prediction data with actual results
    PGA_Prediction <- merge(
      PGA_Prediction,
      PGA_projection[, c("eventID", "playerID", "Top20_Odds", "posn")],
      by = c("eventID", "playerID"),
      all.x = TRUE
    )
    
    log_message(paste("Joined prediction data has", nrow(PGA_Prediction), "rows"))
    
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
      # Determine threshold value - safely handle edge cases
      if (length(PGA_Prediction$Model_Score) >= t) {
        threshold_value <- sort(PGA_Prediction$Model_Score, decreasing = TRUE)[min(t, length(PGA_Prediction$Model_Score))]
      } else {
        log_message(paste("WARNING: Not enough predictions for threshold", t))
        threshold_value <- min(PGA_Prediction$Model_Score)
      }
      
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
  }, error = function(e) {
    log_message(paste("ERROR in prediction processing:", conditionMessage(e)))
    return(NULL)
  })
  
  # Function to create column name for summary metrics
  col_name <- function(metric, threshold, stake_type = NULL) {
    if (is.null(stake_type)) {
      return(paste0(metric, "_top", threshold))
    } else {
      return(paste0(metric, "_", stake_type, "_top", threshold))
    }
  }
  
  # Calculate event summary
  log_message("Calculating event summary")
  tryCatch({
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
      total_bets <- sum(valid_data[[model_score_col]] == 1, na.rm = TRUE)
      successful_bets <- sum(valid_data[[model_score_col]] == 1 & valid_data[[correct_col]] == 1, na.rm = TRUE)
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
  }, error = function(e) {
    log_message(paste("ERROR in event summary calculation:", conditionMessage(e)))
    return(NULL)
  })
  
  # Calculate run metrics for this event
  log_message("Calculating run metrics")
  run_metrics <- data.frame()
  
  tryCatch({
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
  }, error = function(e) {
    log_message(paste("ERROR in run metrics calculation:", conditionMessage(e)))
    return(NULL)
  })
  
  log_message("Evaluation complete")
  # Final garbage collection before returning
  gc(reset = TRUE)
  
  return(run_metrics)
}

# Get all filtered event IDs
pgalist <- split(df, df$eventID)
pgalist_filtered <- pgalist[sapply(pgalist, function(df) nrow(df) > 40)]
all_event_ids <- names(pgalist_filtered) # Adjust this range as needed

# Verify valid event IDs before processing
valid_event_ids <- all_event_ids[sapply(all_event_ids, function(id) {
  if (!id %in% names(pgalist_filtered)) {
    cat("WARNING: Event ID", id, "not in filtered list\n")
    return(FALSE)
  }
  if (nrow(pgalist_filtered[[id]]) < 40) {
    cat("WARNING: Event ID", id, "has fewer than 40 rows\n")
    return(FALSE)
  }
  return(TRUE)
})]

# Use only valid events
cat("Starting with", length(all_event_ids), "events, using", length(valid_event_ids), "valid events\n")

# Create a file name for results with timestamp
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_file <- paste0("model_evaluation_metrics_parallel_", timestamp, ".csv")

# Make sure the log file exists at the start
main_log_file <- "model_progress.log"
writeLines(c("Model run started at", as.character(Sys.time())), main_log_file)

# Lower the core count for stability
cores_to_use <- 12  # Start with just 2 cores to test
cat("Using", cores_to_use, "cores for parallel processing\n")

# Set up parallel processing
cl <- makeCluster(cores_to_use, type = "SOCK")
registerDoParallel(cl)

# Process in smaller batches for better control
batch_size <- 12
all_results <- list()

for (i in seq(1, length(valid_event_ids), by = batch_size)) {
  batch_end <- min(i + batch_size - 1, length(valid_event_ids))
  batch_ids <- valid_event_ids[i:batch_end]
  cat("Processing batch", ceiling(i/batch_size), "with IDs:", paste(batch_ids, collapse=", "), "\n")
  
  # Log batch start
  cat(paste0(Sys.time(), " - Starting batch ", ceiling(i/batch_size), 
             " with events: ", paste(batch_ids, collapse=", "), "\n"), 
      file = main_log_file, append = TRUE)
  
  batch_results <- foreach(
    test_event_id = batch_ids,
    .packages = c("dplyr", "biomod2", "data.table", "rpart", "corrplot", "openxlsx"),
    .combine = "rbind",
    .errorhandling = "pass",
    .export = c("evaluate_model", "%not in%", "df", "eventQuality", 
                "fixed_stake_amount", "target_profit")
  ) %dopar% {
    tryCatch({
      result <- evaluate_model(test_event_id)
      if (is.null(result)) {
        cat(paste0(Sys.time(), " - Event ", test_event_id, ": Null result returned\n"), 
            file = main_log_file, append = TRUE)
      } else {
        cat(paste0(Sys.time(), " - Event ", test_event_id, ": Completed successfully\n"), 
            file = main_log_file, append = TRUE)
      }
      return(result)
    }, error = function(e) {
      cat(paste0(Sys.time(), " - Event ", test_event_id, ": Error: ", conditionMessage(e), "\n"), 
          file = main_log_file, append = TRUE)
      return(NULL)
    })
  }
  
  # Save batch results
  if (!is.null(batch_results) && nrow(batch_results) > 0) {
    batch_file <- paste0("A:/OneDrive - University of Southampton/Golf/temp_status/batch_results_", i, "_", timestamp, ".csv")
    write.csv(batch_results, batch_file, row.names = FALSE)
    cat("Saved batch results to", batch_file, "\n")
    
    # Store batch results
    all_results[[length(all_results) + 1]] <- batch_results
  } else {
    cat("No valid results in this batch\n")
  }
  
  # Force garbage collection between batches
  gc(reset = TRUE)
}

# Stop the cluster
stopCluster(cl)

# Combine all batch results
all_model_metrics <- do.call(rbind, all_results)

# Check if we have any results
if (!is.null(all_model_metrics) && nrow(all_model_metrics) > 0) {
  # Save comprehensive results
  write.csv(all_model_metrics, results_file, row.names = FALSE)
  cat("Saved all results to", results_file, "\n")
  
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
  write.csv(all_model_metrics, "A:/OneDrive - University of Southampton/Golf/Betting Simulations/Top 20/PGA_Top20_Under71EQ_LOOCV_Model6_AllEvents.csv", row.names = FALSE)
  write.csv(aggregate_metrics, "A:/OneDrive - University of Southampton/Golf/Betting Simulations/Top 20/PGA_Top20_Under71EQ_LOOCV_Model6_AggregatedEvents.csv", row.names = FALSE)}
  