#### Load Packages ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
library(foreach)
library(doParallel)
library(caret) # For createDataPartition

'%not in%'  <- function(x,table) is.na(match(x,table,nomatch=NA_integer_))

# Define global constants
fixed_stake_amount <- 10

# Set working directory and load data
setwd("A:/OneDrive - University of Southampton/Golf")
df <- read.csv("./Data/PGA_withodds.csv")
df$rating <- as.numeric(df$rating)
df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df <- df[complete.cases(df),]
eventQuality <- read.csv("./Data/FieldQuality.csv")

df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID")) %>%
  filter(Quality <= 70)

# Create a log file
log_file <- paste0("A:/OneDrive - University of Southampton/Golf/temp_status/modeling_run_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log")
cat(paste0(Sys.time(), " - Starting PGA Top 40 analysis\n"), file = log_file)

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

# Define the variables to be used in the model - adjust based on what's available in your data
model_vars <- c("rating", "rating_rank", "current", "current_rank", "diff_from_mean", 
                "diff_from_median", "diff_from_max", "diff_from_min", "rating_normal",
                "Top40_Odds", "Top20_Odds")

# Check which variables actually exist in the data
available_vars <- model_vars[model_vars %in% names(df)]
cat(paste0(Sys.time(), " - Using variables: ", paste(available_vars, collapse=", "), "\n"), 
    file = log_file, append = TRUE)

# Create a dataset with only the needed variables
model_data <- df %>%
  select(playerID, eventID, posn, top_40, Top40_Odds, all_of(available_vars))

# Split data into training (75%) and testing (25%) sets
set.seed(123) # For reproducibility
train_indices <- createDataPartition(model_data$top_40, p = 0.75, list = FALSE)
train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

cat(paste0(Sys.time(), " - Split data into ", nrow(train_data), " training samples and ", 
           nrow(test_data), " testing samples\n"), file = log_file, append = TRUE)

# Set up model variables
resp <- train_data[, "top_40"]
expl <- train_data[, available_vars]

# Test set preparation
test_resp <- test_data[, c("playerID", "eventID", "top_40", "Top40_Odds")]
test_expl <- test_data[, available_vars]

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
    outerPIsteps = 0, 
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

myOpt <- bm_ModelingOptions(
  data.type = 'binary',
  models = c("ANN", "CTA", "FDA", "GAM", "GBM", "GLM", "MARS", "MAXNET", "RF", "SRE", "XGBOOST"),
  strategy = "user.defined",
  user.val = user.val
)

# Create Biomod Data Object
cat(paste0(Sys.time(), " - Creating Biomod Data Object\n"), file = log_file, append = TRUE)

# Create Biomod Data Object
myBiomodData <- BIOMOD_FormatingData(
  resp.var = resp,
  expl.var = expl,
  resp.name = "PGA")

# Add dummy coordinates (sequential numbers)
dummy_coords <- data.frame(
  x = 1:length(myBiomodData@data.species),
  y = 1:length(myBiomodData@data.species)
)

# Update biomod data object with these coordinates
myBiomodData@coord <- dummy_coords

# Build models
cat(paste0(Sys.time(), " - Building models\n"), file = log_file, append = TRUE)
myBiomodModelOut <- BIOMOD_Modeling(
  bm.format = myBiomodData,
  models = models.to.proc,
  OPT.strategy = "bigboss",
  OPT.user = myOpt,
  OPT.user.val = user.val,
  CV.nb.rep = reps, 
  CV.perc = 0.75,   
  weights = NULL,
  var.import = 3,
  metric.eval = c('TSS', 'ROC'),
  modeling.id = "model_PGA_Top40",
  nb.cpu = 1,
  do.progress = F
)

# Evaluate models
cat(paste0(Sys.time(), " - Evaluating models\n"), file = log_file, append = TRUE)
myBiomodModelEval <- get_evaluations(myBiomodModelOut)

# Create TSS scores table
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

# Create ensemble model
cat(paste0(Sys.time(), " - Creating ensemble model\n"), file = log_file, append = TRUE)
myBiomodEM <- BIOMOD_EnsembleModeling(
  bm.mod = myBiomodModelOut,
  models.chosen = allRun.model.names,
  em.algo = c('EMwmean'),
  metric.select = c("user.defined"),
  metric.select.thresh = c(0.2), 
  metric.select.table = df.metrics,
  metric.eval = c('TSS', 'ROC'),
  var.import = 3,
  EMwmean.decay = 'proportional'
)

# Project ensemble model to test data
cat(paste0(Sys.time(), " - Projecting ensemble model to test data\n"), file = log_file, append = TRUE)
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

# Generate threshold ranges - This is looking at different player ranking thresholds
cat(paste0(Sys.time(), " - Generating threshold ranges\n"), file = log_file, append = TRUE)
threshold_ranges <- list()
counter <- 1

# Generate ranges with increment of 5 (1-5, 6-10, 11-15, etc.)
for (start_val in seq(1, 56, by = 5)) {
  end_val <- start_val + 4
  if (end_val <= 60) {
    threshold_ranges[[counter]] <- list(start = start_val, end = end_val)
    counter <- counter + 1
  }
}

# Generate ranges with increment of 10 (1-10, 11-20, etc.)
for (start_val in seq(1, 51, by = 10)) {
  end_val <- start_val + 9
  if (end_val <= 60) {
    threshold_ranges[[counter]] <- list(start = start_val, end = end_val)
    counter <- counter + 1
  }
}

# Add some additional ranges
additional_ranges <- list(
  list(start = 1, end = 15),
  list(start = 1, end = 25),
  list(start = 1, end = 40),
  list(start = 5, end = 15),
  list(start = 10, end = 25)
)

for (range in additional_ranges) {
  # Check if this range already exists
  exists <- FALSE
  for (i in 1:length(threshold_ranges)) {
    if (threshold_ranges[[i]]$start == range$start && 
        threshold_ranges[[i]]$end == range$end) {
      exists <- TRUE
      break
    }
  }
  
  if (!exists) {
    threshold_ranges[[counter]] <- range
    counter <- counter + 1
  }
}

# Format prediction data
cat(paste0(Sys.time(), " - Processing prediction results\n"), file = log_file, append = TRUE)
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

# Create an event list to process the betting outcomes by event
event_list <- unique(PGA_Prediction$eventID)
cat(paste0(Sys.time(), " - Analyzing ", length(event_list), " events in test data\n"), 
    file = log_file, append = TRUE)

# Initialize data frames to store the results
all_event_results <- data.frame()

# Process each event separately
for (current_event in event_list) {
  # Filter to current event
  event_data <- PGA_Prediction[PGA_Prediction$eventID == current_event, ]
  
  # Calculate for each threshold range
  for (range_idx in 1:length(threshold_ranges)) {
    range <- threshold_ranges[[range_idx]]
    start_val <- range$start
    end_val <- range$end
    
    # Create a range label for columns
    range_label <- paste0(start_val, "to", end_val)
    
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
    
    # Identify players to bet on (those ranked within the threshold range by model score)
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
      EventID = current_event,
      RangeID = paste0("R", range$start, "_", range$end),
      Range_Start = range$start,
      Range_End = range$end,
      Range_Size = range$end - range$start + 1,
      Num_Bets = total_bets,
      Successful_Bets = successful_bets,
      Success_Rate = success_rate,
      Total_Stake = round(total_stake, 2),
      Total_Winnings = round(total_winnings, 2),
      Fixed_Profit = round(total_profit, 2),
      Fixed_ROI = roi,
      stringsAsFactors = FALSE
    )
    
    # Add to all event results
    all_event_results <- rbind(all_event_results, event_range_record)
  }
}

# Calculate aggregate metrics
cat(paste0(Sys.time(), " - Calculating aggregate metrics\n"), file = log_file, append = TRUE)

# Group by range and calculate aggregates
aggregate_metrics <- all_event_results %>%
  group_by(RangeID, Range_Start, Range_End, Range_Size) %>%
  summarize(
    RangeDescription = paste0("Players ranked ", first(Range_Start), "-", first(Range_End)),
    Total_Events = n(),
    Total_Bets = sum(Num_Bets),
    Avg_Num_Bets = mean(Num_Bets),
    Total_Fixed_Profit = sum(Fixed_Profit),
    Avg_Fixed_ROI = mean(Fixed_ROI),
    Fixed_ROI_SD = sd(Fixed_ROI),
    Fixed_ROI_CI_Lower = mean(Fixed_ROI) - 1.96 * sd(Fixed_ROI) / sqrt(n()),
    Fixed_ROI_CI_Upper = mean(Fixed_ROI) + 1.96 * sd(Fixed_ROI) / sqrt(n()),
    Pct_Events_Profitable_Fixed = mean(Fixed_Profit > 0) * 100,
    Risk_Reward_Fixed = ifelse(sd(Fixed_ROI) > 0, mean(Fixed_ROI) / sd(Fixed_ROI), 0),
    .groups = "drop"
  ) %>%
  arrange(desc(Avg_Fixed_ROI))

# Identify top ranges by different metrics
cat(paste0(Sys.time(), " - Identifying top ranges\n"), file = log_file, append = TRUE)

# By Fixed ROI
top_fixed_roi <- aggregate_metrics %>%
  arrange(desc(Avg_Fixed_ROI)) %>%
  head(10)

# By Risk-Adjusted ROI (Sharpe-like ratio)
top_risk_adjusted <- aggregate_metrics %>%
  arrange(desc(Risk_Reward_Fixed)) %>%
  head(10)

# By Most Consistent (highest percent of profitable events)
top_consistency <- aggregate_metrics %>%
  filter(Total_Events >= max(Total_Events) * 0.5) %>% # Only consider ranges with enough events
  arrange(desc(Pct_Events_Profitable_Fixed)) %>%
  head(10)

# Create an executive summary
summary_data <- data.frame(
  Section = c(
    "Overview",
    "Best Overall Range by ROI",
    "Most Consistent Range",
    "Best Risk-Adjusted Range",
    "Recommendation"
  ),
  Content = c(
    paste0("Analysis of ", length(event_list), " different events across ", 
           length(threshold_ranges), " different player ranking thresholds."),
    
    paste0(top_fixed_roi$RangeDescription[1], " (", round(top_fixed_roi$Avg_Fixed_ROI[1], 2), 
           "% ROI, profitable in ", round(top_fixed_roi$Pct_Events_Profitable_Fixed[1], 1), "% of events)"),
    
    paste0(top_consistency$RangeDescription[1], " (profitable in ", 
           round(top_consistency$Pct_Events_Profitable_Fixed[1], 1), "% of events, ", 
           round(top_consistency$Avg_Fixed_ROI[1], 2), "% avg ROI)"),
    
    paste0(top_risk_adjusted$RangeDescription[1], " (", 
           round(top_risk_adjusted$Avg_Fixed_ROI[1], 2), "% ROI with SD of ", 
           round(top_risk_adjusted$Fixed_ROI_SD[1], 2), ")"),
    
    paste0("Based on all metrics, we recommend focusing on the ", 
           top_fixed_roi$RangeDescription[1], " player ranking threshold")
  )
)

# Create Excel report
cat(paste0(Sys.time(), " - Creating Excel report\n"), file = log_file, append = TRUE)
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_wb <- createWorkbook()
text_style <- createStyle(numFmt = "@")  # Text format to prevent Excel date conversion

# Add event results
addWorksheet(results_wb, "All_Event_Results")
writeData(results_wb, "All_Event_Results", all_event_results, keepNA = TRUE)

# Add aggregate metrics
addWorksheet(results_wb, "Aggregate_Metrics")
writeData(results_wb, "Aggregate_Metrics", aggregate_metrics, keepNA = TRUE)

# Add top ranges by different metrics
addWorksheet(results_wb, "Top_Ranges_By_ROI")
writeData(results_wb, "Top_Ranges_By_ROI", top_fixed_roi, keepNA = TRUE)

addWorksheet(results_wb, "Top_Ranges_Risk_Adjusted")
writeData(results_wb, "Top_Ranges_Risk_Adjusted", top_risk_adjusted, keepNA = TRUE)

addWorksheet(results_wb, "Top_Ranges_Consistency")
writeData(results_wb, "Top_Ranges_Consistency", top_consistency, keepNA = TRUE)

# Add executive summary
addWorksheet(results_wb, "Executive_Summary")
writeData(results_wb, "Executive_Summary", summary_data, keepNA = TRUE)

# Set column widths for better readability
setColWidths(results_wb, "Executive_Summary", cols = 1, widths = 25)
setColWidths(results_wb, "Executive_Summary", cols = 2, widths = 100)

# Format all worksheets for better readability
for (sheet in getSheetNames(results_wb)) {
  setColWidths(results_wb, sheet, cols = 1:50, widths = "auto")
}

# Add styles to protect RangeID from date conversion
for (sheet in getSheetNames(results_wb)) {
  sheet_data <- read.xlsx(results_wb, sheet)
  if ("RangeID" %in% colnames(sheet_data)) {
    range_col_idx <- which(colnames(sheet_data) == "RangeID")
    addStyle(results_wb, sheet, style = text_style, 
             rows = 2:(nrow(sheet_data) + 1), 
             cols = range_col_idx, 
             gridExpand = TRUE)
  }
}

# Save workbook with explicit formats to prevent date conversion
options(openxlsx.dateFormat = "yyyy-mm-dd")  # Set global date format
excel_file_path <- paste0("A:/OneDrive - University of Southampton/Golf/Betting Simulations/Top 40/PGA_Top40_Results_", timestamp, ".xlsx")
saveWorkbook(results_wb, excel_file_path, overwrite = TRUE)

# Print summary report to console
cat("\n\n========== ANALYSIS SUMMARY ==========\n")
cat("Total events analyzed:", length(event_list), "\n")
cat("Total player ranking thresholds tested:", length(threshold_ranges), "\n\n")

cat("TOP 5 PLAYER RANKING THRESHOLDS BY ROI:\n")
top5_roi <- aggregate_metrics %>% arrange(desc(Avg_Fixed_ROI)) %>% head(5)
for (i in 1:nrow(top5_roi)) {
  cat(i, ". Players ranked ", top5_roi$Range_Start[i], "-", top5_roi$Range_End[i], " - ", 
      round(top5_roi$Avg_Fixed_ROI[i], 2), "% ROI (", 
      round(top5_roi$Pct_Events_Profitable_Fixed[i], 1), "% profitable events)\n", sep="")
}

cat("\nMOST CONSISTENT THRESHOLD:\n")
top1_consistency <- top_consistency %>% head(1)
cat("Players ranked ", top1_consistency$Range_Start, "-", top1_consistency$Range_End, " - Profitable in ", 
    round(top1_consistency$Pct_Events_Profitable_Fixed, 1), "% of events (", 
    round(top1_consistency$Avg_Fixed_ROI, 2), "% avg ROI)\n", sep="")

cat("\nBEST RISK-ADJUSTED THRESHOLD:\n")
top1_risk <- top_risk_adjusted %>% head(1)
cat("Players ranked ", top1_risk$Range_Start, "-", top1_risk$Range_End, " - ", 
    round(top1_risk$Avg_Fixed_ROI, 2), "% ROI with SD of ", 
    round(top1_risk$Fixed_ROI_SD, 2), " (Risk/Reward: ", round(top1_risk$Risk_Reward_Fixed, 2), ")\n", sep="")

cat("\nAnalysis complete. Results saved to Excel file:", excel_file_path, "\n")