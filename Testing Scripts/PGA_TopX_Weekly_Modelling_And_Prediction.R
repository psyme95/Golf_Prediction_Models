# ===== CONFIGURATION SECTION =====
# Set working directory and parameters
setwd("C:/Projects/Golf/Weekly_Modelling")
set.seed(42)

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'RF', 'ANN', 'GBM', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.7
MODEL_NAME <- paste0("Top20_Prediction_", format(Sys.time(), "%d%m"))

# Training parameters
TRAINING_WINDOW_SIZE <- 80  # Number of most recent events to use for training

# File paths - UPDATE THESE TO YOUR ACTUAL FILE PATHS
HISTORICAL_DATA_FILE <- "./Input/PGA_Processed.xlsx"
UPCOMING_EVENT_FILE <- "./Input/This_Week_Processed.xlsx"

# ===== HELPER FUNCTIONS =====
# Model training function (unchanged from original)
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

# Function to get the most recent N events for training
get_training_data <- function(df_historical, window_size, available_vars) {
  # Get unique events sorted by date
  unique_events <- df_historical %>%
    select(eventID, Date) %>%
    distinct() %>%
    arrange(Date)
  
  # Get the most recent N events
  total_events <- nrow(unique_events)
  recent_events <- tail(unique_events, window_size)
  
  cat("Training on", nrow(recent_events), "most recent events out of", total_events, "total events\n")
  cat("Date range:", min(recent_events$Date), "to", max(recent_events$Date), "\n")
  
  # Extract training data for these events
  train_data <- df_historical %>%
    filter(eventID %in% recent_events$eventID) %>%
    select(playerID, eventID, posn, top_20, Top20_odds, all_of(available_vars))
  
  return(list(train_data = train_data, training_events = recent_events))
}

# ===== LOAD REQUIRED PACKAGES =====
library(biomod2)
library(dplyr)
library(readxl)
library(lubridate)
library(ggplot2)

# Define 'not in' operator
`%not in%` <- Negate(`%in%`)

# ===== DATA LOADING =====
# Load historical events data
df_historical <- read_excel(HISTORICAL_DATA_FILE)
df_historical <- df_historical[complete.cases(df_historical),]

# Load upcoming event data
df_upcoming <- read_excel(UPCOMING_EVENT_FILE)
df_upcoming <- df_upcoming[complete.cases(df_upcoming),]

# Get upcoming event details
cat("Historical events loaded:", length(unique(df_historical$eventID)), "\n")
cat("Upcoming event field size:", nrow(df_upcoming), "\n")

# ===== VARIABLE SETUP =====
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

# Check available variables in both datasets
missing_vars_hist <- model_vars[model_vars %not in% names(df_historical)]
missing_vars_upcoming <- model_vars[model_vars %not in% names(df_upcoming)]
available_vars <- model_vars[model_vars %in% names(df_historical) & model_vars %in% names(df_upcoming)]

cat("Available variables:", length(available_vars), "/", length(model_vars), "\n")
if (length(missing_vars_hist) > 0) {
  cat("Missing in historical data:", paste(missing_vars_hist, collapse = ", "), "\n")
}
if (length(missing_vars_upcoming) > 0) {
  cat("Missing in upcoming event:", paste(missing_vars_upcoming, collapse = ", "), "\n")
}

# ===== MODEL TRAINING =====
# Get training data from most recent events
training_result <- get_training_data(df_historical, TRAINING_WINDOW_SIZE, available_vars)
train_data <- training_result$train_data
training_events <- training_result$training_events

# Train the ensemble model
trained_model <- train_ensemble_model(train_data, available_vars)

# Train calibration model
logit_model_data <- data.frame(
  EventID = train_data$eventID,
  PlayerID = train_data$playerID,
  Model_Score = trained_model@models.prediction@val$pred,
  Actual_Top20 = train_data$top_20,
  Top20_odds = train_data$Top20_odds,
  Top20_ImpliedProb = 1 / train_data$Top20_odds
)

calibration_model <- glm(Actual_Top20 ~ Model_Score + Top20_odds,
                         data = logit_model_data,
                         family = binomial())

print(summary(calibration_model))

# ===== PREDICTION PHASE =====
# Make predictions on upcoming event
test_columns <- c("surname", "firstname", "Top20_odds", "rating", available_vars)
test_columns <- unique(test_columns)
available_test_columns <- test_columns[test_columns %in% names(df_upcoming)]

prediction_data <- df_upcoming %>%
  select(all_of(available_test_columns))

# Prepare data for biomod prediction
test_expl <- prediction_data[, available_vars]
test_resp.xy <- prediction_data[, c("surname", "firstname")]

# Run biomod projection
myBiomodProj <- BIOMOD_EnsembleForecasting(
  bm.em = trained_model,
  bm.proj = NULL,
  proj.name = MODEL_NAME,
  new.env = test_expl,
  new.env.xy = test_resp.xy,
  models.chosen = "all",
  metric.binary = "TSS",
  metric.filter = "TSS",
  na.rm = TRUE
)

# Process predictions
predictions <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
colnames(predictions)[3] <- "Model_Score"

# Merge with original data
essential_cols <- c("surname", "firstname", "Top20_odds", "rating")
merge_cols <- intersect(names(prediction_data), essential_cols)

predictions <- merge(
  predictions,
  prediction_data[, merge_cols],
  by = c("surname", "firstname"), all.x = TRUE
)

# Apply calibration
calibration_data <- data.frame(
  Model_Score = predictions$Model_Score,
  Top20_odds = predictions$Top20_odds
)

predictions$Calibrated_Probability <- predict(calibration_model, 
                                              calibration_data, 
                                              type = "response")

# Create final prediction results
final_predictions <- data.frame(
  Surname = predictions$surname,
  Firstname = predictions$firstname,
  Rating = predictions$rating,
  Model_Score = round(predictions$Model_Score, 4),
  Top20_odds = predictions$Top20_odds,
  Calibrated_Probability = round(predictions$Calibrated_Probability, 4),
  Calibrated_Odds = round(1 / predictions$Calibrated_Probability, 2)
) %>%
  arrange(Top20_odds) %>%
  mutate(Lay_Odds = Top20_odds * 1.1,
         Value_Bet = ifelse(Calibrated_Odds > Lay_Odds & Lay_Odds < 1000, 1, 0),
  )

# ===== RESULTS AND EXPORT =====
cat("\n===== RESULTS SUMMARY =====\n")
cat("Field Size:", nrow(final_predictions), "\n")
cat("Training Period:", min(training_events$eventID), "to", max(training_events$eventID), "\n")
cat("Number of Training Events:", nrow(training_events), "\n")

# Create results directory
results_dir <- paste0("./Output/")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
}

# Save predictions
prediction_file <- paste0(results_dir, MODEL_NAME, ".xlsx")
write.xlsx(final_predictions, prediction_file, rowNames = FALSE)
