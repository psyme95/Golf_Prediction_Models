#### Load Packages ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
library(foreach)
library(doSNOW)  # Changed from doParallel for progress bar support
library(caret)

'%not in%'  <- function(x,table) is.na(match(x,table,nomatch=NA_integer_))

# Define global constants
fixed_stake_amount <- 10

# Set working directory and load data
setwd("C:/Projects/Golf")
df <- read.csv("./Data/PGA_withodds.csv")
df$rating <- as.numeric(df$rating)
df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df <- df[complete.cases(df),]
eventQuality <- read.csv("./Data/FieldQuality.csv")

df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID")) %>%
  filter(Quality >= 1 & Quality <= 700)

# Create a log file
log_file <- paste0("C:/Projects/Golf/temp_status/modeling_run_LOOCV_", format(Sys.time(), "%m%d_%H%M"), ".log")
cat(paste0(Sys.time(), " - Starting PGA Top 40 LOOCV analysis\n"), file = log_file)

# Create derived features
cat(paste0(Sys.time(), " - Creating derived features\n"), file = log_file, append = TRUE)
df$top_40 <- ifelse(df$posn <= 40, 1, 0)

# Split dataframe by eventID to calculate tournament-specific metrics
events <- split(df, df$eventID)
df_with_diffs <- data.frame()

# Process each tournament separately
for (event_id in names(events)) {
  event_data <- events[[event_id]]
  
  # Calculate tournament-specific statistics
  event_mean <- mean(event_data$rating, na.rm = TRUE)
  event_median <- median(event_data$rating, na.rm = TRUE)
  event_max <- max(event_data$rating, na.rm = TRUE)
  event_min <- min(event_data$rating, na.rm = TRUE)
  
  # Add difference variables for this event
  event_data$diff_from_mean <- event_data$rating - event_mean
  event_data$diff_from_median <- event_data$rating - event_median
  event_data$diff_from_max <- event_data$rating - event_max
  event_data$diff_from_min <- event_data$rating - event_min
  
  # Add normalized ratings within this event
  event_data$rating_normal <- scale(event_data$rating)
  
  # Add to the combined dataframe
  df_with_diffs <- rbind(df_with_diffs, event_data)
}

# Replace original dataframe with the one containing tournament-specific differences
df <- df_with_diffs

# Check for any NA values in important columns
na_count <- sum(!complete.cases(df[, c("top_40", "rating", "Top40_Odds")]))
if (na_count > 0) {
  cat(paste0(Sys.time(), " - WARNING: Found ", na_count, " rows with NA values. Removing these rows.\n"), 
      file = log_file, append = TRUE)
  df <- df[complete.cases(df[, c("top_40", "rating", "Top40_Odds")]), ]
}

# Define the variables to be used in the model
run <- "2d"
model_vars <- c("log_rating", "current_rank", "compat_rank", "Top40_Odds", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank","sgapp", "Starts_Not10")

# Check which variables actually exist in the data
available_vars <- model_vars[model_vars %in% names(df)]
cat(paste0(Sys.time(), " - Using variables: ", paste(available_vars, collapse=", "), "\n"), 
    file = log_file, append = TRUE)

# Create a dataset with only the needed variables
model_data <- df %>%
  select(playerID, eventID, posn, top_40, all_of(available_vars))

##### Variable Correlation Check ####
cor_results <- cor(model_data[,-c(1:4)],
                   use = "pairwise.complete.obs")

corrplot(cor_results, method = "number")

# Get unique event IDs for LOOCV
unique_events <- unique(model_data$eventID)
cat(paste0(Sys.time(), " - Starting LOOCV with ", length(unique_events), " events\n"), 
    file = log_file, append = TRUE)

# Generate threshold ranges systematically
ranges_df <- data.frame()
size_increments <- c(5, 10, 15, 20, 25, 30, 35, 40)

for (size in size_increments) {
  max_start <- 61 - size
  for (start_val in seq(1, max_start, by = 5)) {
    end_val <- start_val + size - 1
    ranges_df <- rbind(ranges_df, data.frame(start = start_val, end = end_val))
  }
}

ranges_df <- unique(ranges_df)
threshold_ranges <- list()
for (i in 1:nrow(ranges_df)) {
  threshold_ranges[[i]] <- list(
    start = ranges_df$start[i], 
    end = ranges_df$end[i]
  )
}

# Initialize results storage
all_event_results <- data.frame()
all_predictions <- data.frame()
parallel_summary <- list()  # Store summary info from parallel processing

# Biomod model parameters
reps <- 1
models.to.proc = c('GAM', 'GLM', 'RF', 'FDA', 'CTA', 'ANN', 'GBM', 'MAXNET', 'XGBOOST')

# Define model parameters
user.rf <- list('_allData_allRun' = list(ntree=500, nodesize=100))
user.XGBOOST <- list('_allData_allRun' = list(nrounds = 10))
user.GAM <- list('_allData_allRun' = list( 
  algo = 'GAM_mgcv',
  type = 's_smoother',
  k = 4,
  interaction.level = 1,
  myFormula = NULL,
  family = binomial(link = 'logit'),
  method = 'GCV.Cp',
  optimizer = c('outer','newton'),
  select = FALSE,
  knots = NULL,
  paraPen = NULL,
  control = list(
    nthreads = 1, irls.reg = 0, epsilon = 1e-07, maxit = 200, 
    trace = FALSE, mgcv.tol = 1e-07, mgcv.half = 15,
    rank.tol = 1.49011611938477e-08, 
    nlm = list(ndigit=7, gradtol=1e-06, stepmax=2, steptol=1e-04, iterlim=200, check.analyticals=0), 
    optim = list(factr=1e+07), 
    newton = list(conv.tol=1e-06, maxNstep=5, maxSstep=2, maxHalf=30, use.svd=0),
    idLinksBases = TRUE, scalePenalty = TRUE, keepData = FALSE
  )
))
user.GLM <- list('_allData_allRun' = list(type = "quadratic", interaction.level=0))
user.SRE <- list('_allData_allRun' = list(quant=0.025))
user.MARS <- list('_allData_allRun' = list(type='quadratic', interaction.level = 0))
user.CTA <- list('_allData_allRun' = list(control = rpart.control(cp=0.005)))

user.val <- list(
  RF.binary.randomForest.randomForest = user.rf,
  XGBOOST.binary.xgboost.xgboost = user.XGBOOST,
  GAM.binary.mgcv.gam = user.GAM,
  GLM.binary.stats.glm = user.GLM,
  SRE.binary.biomod2.bm_SRE = user.SRE,
  MARS.binary.earth.earth = user.MARS,
  CTA.binary.rpart.rpart = user.CTA
)

# Set up parallel processing with doSNOW for progress bar support
n_cores <- 12
cl <- makeCluster(n_cores, type = "SOCK")
registerDoSNOW(cl)

cat(paste0(Sys.time(), " - Setting up parallel processing with ", n_cores, " cores\n"), 
    file = log_file, append = TRUE)

# Create a progress tracking function that correctly handles completed events
completed_events <- 0
start_time_overall <- Sys.time()

# Simple progress function - just updates the progress bar
progress <- function(n) {
  completed_events <<- completed_events + 1
  setTxtProgressBar(pb, completed_events)
  
  # Log basic progress without trying to guess which event
  elapsed_time <- as.numeric(difftime(Sys.time(), start_time_overall, units = "mins"))
  if (completed_events > 1) {
    avg_time_per_event <- elapsed_time / completed_events
    remaining_events <- length(unique_events) - completed_events
    eta_minutes <- avg_time_per_event * remaining_events
    eta_time <- Sys.time() + (eta_minutes * 60)
    
    cat(paste0(Sys.time(), " - Progress: ", completed_events, "/", length(unique_events), 
               " events completed | Elapsed: ", round(elapsed_time, 1), 
               " mins | ETA: ", format(eta_time, "%H:%M:%S"), "\n"))
  } else {
    cat(paste0(Sys.time(), " - Progress: ", completed_events, "/", length(unique_events), 
               " events completed\n"))
  }
  
  flush.console()
}

# Create progress bar
pb <- txtProgressBar(min = 0, max = length(unique_events), style = 3)
opts <- list(progress = progress)

# Export necessary objects and functions to workers
clusterExport(cl, c("model_data", "available_vars", "fixed_stake_amount", 
                    "threshold_ranges", "models.to.proc", "user.val", "reps",
                    "user.rf", "user.XGBOOST", "user.GAM", "user.GLM", 
                    "user.SRE", "user.MARS", "user.CTA", "log_file"))

# Load required packages on each worker
clusterEvalQ(cl, {
  library(biomod2)
  library(dplyr)
  library(data.table)
  library(rpart)
})

cat(paste0(Sys.time(), " - Starting parallel LOOCV with corrected progress tracking\n"), 
    file = log_file, append = TRUE)
cat("Starting parallel LOOCV processing...\n")

# Fixed parallel LOOCV Loop 
loocv_results <- foreach(i = 1:length(unique_events), 
                         .combine = function(x, y) {
                           list(
                             event_results = rbind(x$event_results, y$event_results),
                             predictions = rbind(x$predictions, y$predictions),
                             summary_info = rbind(x$summary_info, y$summary_info)
                           )
                         },
                         .init = list(event_results = data.frame(), 
                                      predictions = data.frame(),
                                      summary_info = data.frame()),
                         .packages = c('biomod2', 'dplyr', 'data.table', 'rpart'),
                         .options.snow = opts,
                         .errorhandling = 'pass',
                         .verbose = FALSE) %dopar% {
                           
                           # Get event info for this worker
                           test_event <- unique_events[i]
                           worker_start_time <- Sys.time()
                           
                           # Log the start of processing for this specific event
                           tryCatch({
                             cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                        " STARTED: EventID ", test_event, "\n"), 
                                 file = log_file, append = TRUE)
                           }, error = function(e) {
                             # Ignore logging errors to prevent worker crashes
                           })
                           
                           # Wrap entire worker code in outer try-catch
                           outer_result <- tryCatch({
                             
                             train_events <- unique_events[-i]
                             
                             # Split data for this fold
                             train_data <- model_data[model_data$eventID %in% train_events, ]
                             test_data <- model_data[model_data$eventID == test_event, ]
                             
                             # Initialize return objects for this worker
                             worker_event_results <- data.frame()
                             worker_predictions <- data.frame()
                             worker_summary <- data.frame()
                             
                             # Track processing info
                             status <- "processing"
                             error_msg <- NA
                             
                             # Check if test event has too few observations
                             if (nrow(test_data) < 40) {
                               # Log completion to file
                               processing_time <- as.numeric(difftime(Sys.time(), worker_start_time, units = "secs"))
                               tryCatch({
                                 cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                            " COMPLETED: EventID ", test_event, 
                                            " - SKIPPED (", nrow(test_data), " players < 80) - ", 
                                            round(processing_time, 1), " seconds\n"), 
                                     file = log_file, append = TRUE)
                               }, error = function(e) {})
                               
                               worker_summary <- data.frame(
                                 eventID = test_event,
                                 status = "skipped",
                                 reason = "too few observations",
                                 n_obs = nrow(test_data),
                                 processing_time = processing_time,
                                 error = NA,
                                 stringsAsFactors = FALSE
                               )
                               
                               return(list(
                                 event_results = data.frame(),
                                 predictions = data.frame(),
                                 summary_info = worker_summary
                               ))
                             }
                             
                             # Log that we're proceeding with modeling
                             tryCatch({
                               cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                          " MODELING: EventID ", test_event, 
                                          " (", nrow(test_data), " players)\n"), 
                                   file = log_file, append = TRUE)
                             }, error = function(e) {})
                             
                             # Set up model variables
                             resp <- train_data[, "top_40"]
                             expl <- train_data[, available_vars[-3]]
                             
                             # Test set preparation
                             test_resp <- test_data[, c("playerID", "eventID", "top_40", "Top40_Odds", "posn")]
                             test_expl <- test_data[, available_vars[-3]]
                             
                             tryCatch({
                               # Create Biomod Data Object
                               myBiomodData <- BIOMOD_FormatingData(
                                 resp.var = resp,
                                 expl.var = expl,
                                 resp.name = "PGA")
                               
                               # Add dummy coordinates
                               dummy_coords <- data.frame(
                                 x = 1:length(myBiomodData@data.species),
                                 y = 1:length(myBiomodData@data.species)
                               )
                               myBiomodData@coord <- dummy_coords
                               
                               # Create modeling options
                               myOpt <- bm_ModelingOptions(data.type = 'binary',
                                                           models=c("ANN", "CTA", "FDA", "GAM", "GBM", "GLM", "MARS", "MAXNET", "RF",
                                                                    "SRE", "XGBOOST"),
                                                           strategy = "user.defined",
                                                           user.val = user.val,
                                                           bm.format = myBiomodData)
                               
                               # Build models
                               myBiomodModelOut <- BIOMOD_Modeling(
                                 bm.format = myBiomodData,
                                 models = models.to.proc,
                                 OPT.strategy = "bigboss",
                                 #OPT.user = myOpt,
                                 #OPT.user.val = user.val,
                                 CV.nb.rep = reps, 
                                 CV.perc = 0.75,   
                                 weights = NULL,
                                 var.import = 1,
                                 metric.eval = c('TSS'),
                                 nb.cpu = 1,
                                 do.progress = F
                               )
                               
                               # Log that modeling is complete, now evaluating
                               tryCatch({
                                 cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                            " EVALUATING: EventID ", test_event, "\n"), 
                                     file = log_file, append = TRUE)
                               }, error = function(e) {})
                               
                               # Evaluate models
                               myBiomodModelEval <- get_evaluations(myBiomodModelOut)
                               
                               # Create TSS scores table
                               allRun.model.names <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"full.name"]
                               allRun.model.sens <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"sensitivity"]
                               allRun.model.spec <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"specificity"]
                               allRun.model.TSS <- (allRun.model.sens/100+allRun.model.spec/100)-1
                               
                               df.metrics  <-  data.frame(matrix(0, ncol = length(allRun.model.TSS), nrow = 1))
                               names(df.metrics) <- allRun.model.names
                               row.names(df.metrics) <- "TSS"
                               df.metrics[1,1:length(allRun.model.TSS)] <- allRun.model.TSS
                               
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
                               
                               # Project ensemble model to test data
                               test_resp.xy <- test_resp[, c("playerID", "eventID")]
                               
                               # Ensemble forecasting
                               myBiomodProj <- BIOMOD_EnsembleForecasting(
                                 bm.em = myBiomodEM,
                                 bm.proj = NULL,
                                 proj.name = "test_set",
                                 new.env = test_expl,
                                 new.env.xy = test_resp.xy,
                                 models.chosen = "all",
                                 metric.binary = "TSS",
                                 metric.filter = "TSS",
                                 na.rm = TRUE
                               )
                               
                               # Format prediction data
                               PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
                               colnames(PGA_Prediction)[3] <- "Model_Score"
                               
                               # Join prediction data with actual results
                               PGA_Prediction <- merge(
                                 PGA_Prediction,
                                 test_data[, c("eventID", "playerID", "Top40_Odds", "posn", "top_40")],
                                 by = c("eventID", "playerID"),
                                 all.x = TRUE
                               )
                               
                               # Setup odds validity
                               PGA_Prediction$Valid_Odds <- PGA_Prediction$Top40_Odds > 1
                               PGA_Prediction$Fixed_Stake <- fixed_stake_amount
                               
                               # Store predictions for this event
                               worker_predictions <- PGA_Prediction
                               
                               # Log that predictions are complete, now calculating betting outcomes
                               tryCatch({
                                 cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                            " BETTING CALC: EventID ", test_event, "\n"), 
                                     file = log_file, append = TRUE)
                               }, error = function(e) {})
                               
                               # Process betting outcomes for this event
                               event_data <- PGA_Prediction
                               
                               # Calculate for each threshold range
                               for (range_idx in 1:length(threshold_ranges)) {
                                 range <- threshold_ranges[[range_idx]]
                                 start_val <- range$start
                                 end_val <- range$end
                                 
                                 # Get the total number of players in the event
                                 total_players <- nrow(event_data)
                                 
                                 # Safely determine start and end threshold values based on ranks
                                 sorted_scores <- sort(event_data$Model_Score, decreasing = TRUE)
                                 
                                 # Handle case where there aren't enough players
                                 if (length(sorted_scores) < start_val) {
                                   start_threshold_value <- min(sorted_scores, na.rm = TRUE)
                                 } else {
                                   start_threshold_value <- sorted_scores[min(start_val, total_players)]
                                 }
                                 
                                 if (length(sorted_scores) < end_val) {
                                   end_threshold_value <- min(sorted_scores, na.rm = TRUE)
                                 } else {
                                   end_threshold_value <- sorted_scores[min(end_val, total_players)]
                                 }
                                 
                                 # Identify players to bet on
                                 bet_players <- event_data$Model_Score >= end_threshold_value & 
                                   event_data$Model_Score <= start_threshold_value & 
                                   event_data$Valid_Odds
                                 
                                 # Calculate betting results for this event and range
                                 total_bets <- sum(bet_players, na.rm = TRUE)
                                 total_stake <- total_bets * fixed_stake_amount
                                 successful_bets <- sum(bet_players & event_data$top_40 == 1, na.rm = TRUE)
                                 success_rate <- ifelse(total_bets > 0, round(successful_bets / total_bets * 100, 2), 0)
                                 
                                 # Calculate winnings and profit
                                 total_winnings <- sum(ifelse(bet_players & event_data$top_40 == 1, 
                                                              fixed_stake_amount * event_data$Top40_Odds, 0), na.rm = TRUE)
                                 total_profit <- total_winnings - total_stake
                                 roi <- ifelse(total_stake > 0, round(total_profit / total_stake * 100, 2), 0)
                                 
                                 # Create a record for this event-range combination
                                 event_range_record <- data.frame(
                                   EventID = test_event,
                                   Range_Start = range$start,
                                   Range_End = range$end,
                                   Num_Bets = total_bets,
                                   Successful_Bets = successful_bets,
                                   Success_Rate = success_rate,
                                   Total_Stake = round(total_stake, 2),
                                   Total_Winnings = round(total_winnings, 2),
                                   Fixed_Profit = round(total_profit, 2),
                                   Fixed_ROI = roi,
                                   stringsAsFactors = FALSE
                                 )
                                 
                                 # Add to worker event results
                                 worker_event_results <- rbind(worker_event_results, event_range_record)
                               }
                               
                               status <- "completed"
                               
                             }, error = function(e) {
                               status <- "error"
                               error_msg <- e$message
                               
                               # Log error
                               tryCatch({
                                 cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                            " ERROR: EventID ", test_event, 
                                            " - ", e$message, "\n"), 
                                     file = log_file, append = TRUE)
                               }, error = function(e2) {})
                             })
                             
                             # Create summary record and log final completion
                             end_time <- Sys.time()
                             processing_time <- as.numeric(difftime(end_time, worker_start_time, units = "secs"))
                             
                             # Log final completion status
                             tryCatch({
                               cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                          " COMPLETED: EventID ", test_event, 
                                          " - ", toupper(status), " (", round(processing_time, 1), " seconds)\n"), 
                                   file = log_file, append = TRUE)
                             }, error = function(e) {})
                             
                             worker_summary <- data.frame(
                               eventID = test_event,
                               status = status,
                               reason = ifelse(status == "completed", "success", "modeling error"),
                               n_obs = nrow(test_data),
                               processing_time = processing_time,
                               error = error_msg,
                               stringsAsFactors = FALSE
                             )
                             
                             # Return results for this worker
                             return(list(
                               event_results = worker_event_results,
                               predictions = worker_predictions,
                               summary_info = worker_summary
                             ))
                             
                           }, error = function(e) {
                             # If any error occurs at the top level, log and create error summary
                             processing_time <- as.numeric(difftime(Sys.time(), worker_start_time, units = "secs"))
                             tryCatch({
                               cat(paste0(Sys.time(), " - Event ", i, "/", length(unique_events), 
                                          " COMPLETED: EventID ", test_event, 
                                          " - TOP-LEVEL ERROR (", round(processing_time, 1), " seconds): ", e$message, "\n"), 
                                   file = log_file, append = TRUE)
                             }, error = function(e2) {})
                             
                             error_summary <- data.frame(
                               eventID = test_event,
                               status = "error",
                               reason = "top-level error",
                               n_obs = NA,
                               processing_time = processing_time,
                               error = paste0("Unexpected error: ", e$message),
                               stringsAsFactors = FALSE
                             )
                             
                             return(list(
                               event_results = data.frame(),
                               predictions = data.frame(),
                               summary_info = error_summary
                             ))
                           })
                           
                           return(outer_result)
                         }

# Close progress bar
close(pb)

# Stop the cluster
stopCluster(cl)

cat(paste0(Sys.time(), " - Parallel processing completed\n"), file = log_file, append = TRUE)
cat("Parallel processing completed!\n")

# Extract results from parallel processing
all_event_results <- loocv_results$event_results
all_predictions <- loocv_results$predictions
parallel_summary <- loocv_results$summary_info

# Check for missing events
processed_events <- unique(parallel_summary$eventID)
missing_events <- setdiff(unique_events, processed_events)

if (length(missing_events) > 0) {
  cat(paste0(Sys.time(), " - WARNING: ", length(missing_events), " events missing from results\n"), 
      file = log_file, append = TRUE)
  cat(paste0(Sys.time(), " - Missing events: ", paste(missing_events, collapse = ", "), "\n"), 
      file = log_file, append = TRUE)
  
  # Add missing events to summary with unknown status
  for (event in missing_events) {
    missing_summary <- data.frame(
      eventID = event,
      status = "missing",
      reason = "not returned from parallel processing",
      n_obs = NA,
      processing_time = 0,
      error = "Unknown - event result not captured",
      stringsAsFactors = FALSE
    )
    parallel_summary <- rbind(parallel_summary, missing_summary)
  }
}

# Log summary of parallel processing
cat(paste0(Sys.time(), " - Parallel LOOCV completed\n"), file = log_file, append = TRUE)
cat(paste0(Sys.time(), " - Events processed: ", sum(parallel_summary$status == "completed"), "\n"), 
    file = log_file, append = TRUE)
cat(paste0(Sys.time(), " - Events skipped: ", sum(parallel_summary$status == "skipped"), "\n"), 
    file = log_file, append = TRUE)
cat(paste0(Sys.time(), " - Events with errors: ", sum(parallel_summary$status == "error"), "\n"), 
    file = log_file, append = TRUE)
cat(paste0(Sys.time(), " - Events missing: ", sum(parallel_summary$status == "missing"), "\n"), 
    file = log_file, append = TRUE)
cat(paste0(Sys.time(), " - Total processing time: ", round(sum(parallel_summary$processing_time, na.rm = TRUE), 2), " seconds\n"), 
    file = log_file, append = TRUE)

# Log details of any errors
if (any(parallel_summary$status %in% c("error", "missing"))) {
  error_events <- parallel_summary[parallel_summary$status %in% c("error", "missing"), ]
  cat(paste0(Sys.time(), " - Error/Missing details:\n"), file = log_file, append = TRUE)
  for (i in 1:nrow(error_events)) {
    cat(paste0("    Event ", error_events$eventID[i], " (", error_events$status[i], "): ", 
               error_events$error[i], "\n"), 
        file = log_file, append = TRUE)
  }
}

# Calculate aggregate metrics after LOOCV
cat(paste0(Sys.time(), " - Calculating aggregate metrics from LOOCV\n"), file = log_file, append = TRUE)

# Group by range and calculate aggregates
aggregate_metrics <- all_event_results %>%
  filter(Num_Bets > 0) %>%
  group_by(Range_Start, Range_End) %>%
  summarize(
    Total_Events = n(),
    Total_Bets = sum(Num_Bets),
    Avg_Num_Bets = mean(Num_Bets),
    Total_Fixed_Profit = sum(Fixed_Profit),
    Avg_Fixed_ROI = mean(Fixed_ROI),
    Fixed_ROI_SD = sd(Fixed_ROI),
    Fixed_ROI_CI_Lower = mean(Fixed_ROI) - 1.96 * sd(Fixed_ROI) / sqrt(n()),
    Fixed_ROI_CI_Upper = mean(Fixed_ROI) + 1.96 * sd(Fixed_ROI) / sqrt(n()),
    Pct_Events_Profitable_Fixed = mean(Fixed_Profit > 0) * 100,
    Sharpe_Ratio = ifelse(sd(Fixed_ROI) > 0, mean(Fixed_ROI) / sd(Fixed_ROI), 0),
    .groups = "drop"
  ) %>%
  arrange(desc(Avg_Fixed_ROI))

# Create Excel report
cat(paste0(Sys.time(), " - Creating Excel report for LOOCV results\n"), file = log_file, append = TRUE)
results_wb <- createWorkbook()

# Add event results
all_event_results_clean <- all_event_results %>%
  rename(
    ROI = Fixed_ROI,
    Profit = Fixed_Profit
  )

# Add aggregate metrics
aggregate_metrics_clean <- aggregate_metrics %>%
  rename(
    Avg_ROI = Avg_Fixed_ROI,
    ROI_SD = Fixed_ROI_SD,
    ROI_CI_Lower = Fixed_ROI_CI_Lower,
    ROI_CI_Upper = Fixed_ROI_CI_Upper,
    Total_Profit = Total_Fixed_Profit,
    Pct_Events_Profitable = Pct_Events_Profitable_Fixed,
    Sharpe_Ratio = Sharpe_Ratio
  )

# Add worksheets
addWorksheet(results_wb, "All_Event_Results")
writeData(results_wb, "All_Event_Results", all_event_results_clean, keepNA = TRUE)

addWorksheet(results_wb, "Aggregate_Metrics")
writeData(results_wb, "Aggregate_Metrics", aggregate_metrics_clean, keepNA = TRUE)

addWorksheet(results_wb, "Processing_Summary")
writeData(results_wb, "Processing_Summary", parallel_summary, keepNA = TRUE)

# Save workbook
excel_file_path <- paste0("C:/Projects/Golf/Betting Simulations/Top 40/PGA_Top40_LOOCV_Results_Run", run , ".xlsx")
saveWorkbook(results_wb, excel_file_path, overwrite = TRUE)

cat(paste0(Sys.time(), " - LOOCV Analysis complete. Results saved to: ", excel_file_path, "\n"), 
    file = log_file, append = TRUE)

# Final summary in log
cat("\n===== FINAL SUMMARY =====\n", file = log_file, append = TRUE)
cat(paste0("Total events analyzed: ", length(unique_events), "\n"), file = log_file, append = TRUE)
cat(paste0("Successfully processed: ", sum(parallel_summary$status == "completed"), "\n"), file = log_file, append = TRUE)
cat(paste0("Best performing range: ", aggregate_metrics_clean$Range_Start[1], "-", 
           aggregate_metrics_clean$Range_End[1], " with ROI: ", 
           round(aggregate_metrics_clean$Avg_ROI[1], 2), "%\n"), file = log_file, append = TRUE)
cat(paste0("Log file saved to: ", log_file, "\n"), file = log_file, append = TRUE)
