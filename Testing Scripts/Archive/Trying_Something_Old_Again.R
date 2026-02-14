#### Enhanced Sequential Betting Strategy Simulation - FIXED STAKES ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(openxlsx)
library(ggplot2)
library(gridExtra)
library(readxl)
library(lubridate)

'%not in%' <- function(x, table) is.na(match(x, table, nomatch = NA_integer_))

# ===== CONFIGURATION SECTION =====
# Set working directory and parameters
setwd("C:/Projects/Golf")
set.seed(42)
MODEL_NAME <- paste0("Top40_Yr2023_", format(Sys.time(), "%d%m_%H%M"))

# Simulation parameters - moved to top for easy configuration
STARTING_BANKROLL <- 1000
FIXED_STAKE <- 10  # Fixed £10 per bet (changed from percentage)
MIN_PLAYERS_PER_EVENT <- 1

# Model parameters
MODELS_TO_PROCESS <- c('GAM', 'GLM', 'RF', 'FDA', 'CTA', 'ANN', 'GBM', 'MAXNET', 'XGBOOST')
CV_REPETITIONS <- 1
CV_PERCENTAGE <- 0.75

# ===== HELPER FUNCTIONS =====
# Enhanced feature engineering function
create_derived_features <- function(df, log_file = NULL) {
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
  } else {
    df$performance_trend <- NA
    missing_vars <- c("current", "X_6m", "X_1yr", "yr3_All")[!c("current", "X_6m", "X_1yr", "yr3_All") %in% names(df)]
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
  } else {
    df$course_advantage <- NA
    missing_vars <- c("course", "yr3_All")[!c("course", "yr3_All") %in% names(df)]
  }
  
  # Event-level features with progress tracking
  events <- split(df, df$eventID)
  df_with_diffs <- data.frame()
  

  for (i in seq_along(events)) {
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
    return(df_with_diffs)
}

# ===== MAIN EXECUTION =====
# Load and validate data (KEEP QUALITY DATA FOR INFORMATION PURPOSES)
eventQuality <- read.csv("./Data/FieldQuality.csv")
eventDates <- read.csv("./Data/EventDates.csv")

df_old <- read_excel("./Data/PGA_revised_odds.xlsx")
names(df_old) <- gsub("^_", "X_", names(df_old))
df_old$rating <- as.numeric(df_old$rating)
df_old <- df_old[complete.cases(df_old), ]
df_old$eventID <- as.numeric(df_old$eventID)
df_old <- df_old %>% 
  left_join(eventQuality, by = c("eventID" = "EventID")) %>%
  left_join(eventDates, by = c("eventID" = "EventID"))

df_old <- create_derived_features(df_old, log_file)

# Split events into list elements and randomly select 20% for testing
event_list <- split(df_old, df_old$eventID)
#event_list <- event_list[sapply(event_list, function(df) nrow(df) > 80)]

# selected_indices <- sample(1:length(event_list), round(length(event_list) * 0.3))
# 
# test_event_list <- list()
# for (i in selected_indices) {
#   test_event_list[[length(test_event_list) + 1]] <- event_list[[i]]  # Add to new list
# }

# test_event_list <- event_list[117:156]

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


# Remove testing events from original list
# train_event_list <- event_list[-selected_indices]

##### Combine all tournament data into one dataframe #####
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
model_vars <- c("field", "current_rank", "compat_rank", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "sgapp", "Starts_Not10", "diff_from_max", "current_top20", "compat2",
                "AvPosn_Rank", "yr3_All_rank", "performance_trend", "Top40_Odds")

model_vars <- c("field", "current_rank", "compat_rank", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "sgapp", "Starts_Not10", "diff_from_max", "current_top20", "compat2",
                "AvPosn_Rank", "yr3_All_rank", "Top40_Odds")

# Check available variables
available_vars_old <- model_vars[model_vars %in% names(df_old)]
available_vars_new <- model_vars[model_vars %in% names(df_new)]
available_vars <- intersect(available_vars_old, available_vars_new)

# Create datasets
train_data <- df_old %>%
  select(playerID, eventID, posn, top_40, all_of(available_vars))

test_data <- df_new %>%
  select(playerID, eventID, posn, top_40, Quality, Top40_Profit, rating, all_of(available_vars))

# Train ensemble model
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
  metric.select.thresh = c(0.1),
  metric.select.table = df.metrics,
  metric.eval = c('TSS'),
  var.import = 1,
  EMwmean.decay = 'proportional'
)

#saveRDS(myBiomodEM, file = paste0("./Models/", MODEL_NAME, "_BiomodEM.rds"))

source("C:/Projects/Golf/R Scripts/cutoff_finder_biomod.R")

result <- cutoff_finder_biomod(
  biomod_ensemble = myBiomodEM,
  biomod_data_object = myBiomodData,
  sensitivity_target = 1,
  specificity_target = 1
)

# View full performance metrics
sensspecthreshold <- print(result$full_performance)
sensspecthreshold[,2:4] <- round(sensspecthreshold[,2:4] / 100, 2)

# Testing
test_data
resp_test <- test_data[, "top_40"]
expl_test <- test_data[, model_vars]
xy_test <- test_data[, 1:2]


# Ensemble forecasting remains the same
myBiomodProj <- BIOMOD_EnsembleForecasting(
  bm.em = myBiomodEM,
  bm.proj = NULL,
  proj.name = "wmean",
  new.env = expl_test,
  new.env.xy = xy_test,
  models.chosen = "all",
  metric.binary = "TSS",
  metric.filter = "TSS",
  na.rm = TRUE
)

# Format prediction data
PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
colnames(PGA_Prediction)[3] <- "Model_Score"

projections <- data.frame()

projections <- PGA_Prediction %>%
  left_join(
    test_data %>% select(playerID, eventID, field, Quality, posn, top_40, Top40_Odds, Top40_Profit),
    by = c("playerID", "eventID")
  ) %>%
  left_join(sensspecthreshold, by = c("Model_Score" = "threshold")) %>%
  mutate(Winners = 40,
         Losers = field - Winners,
         TP = round(sensitivity * Winners),
         FP = round((1 - specificity) * Losers),
         min_odds = (TP + FP) / TP,
         odds_diff = Top40_Odds - min_odds,
         good_odds = Top40_Odds > min_odds)

projections %>%
  filter(odds_diff >= 2 * min_odds) %>%
  summarise(sum(Top40_Profit))

hist(projections$Model_Score, breaks = 30)
ks.test(projections$Model_Score, 'pnorm')

qqnorm(projections$Model_Score, main='Normal')
qqline(projections$Model_Score)

qqnorm(log(projections$Model_Score), main='Log')
qqline(log(projections$Model_Score))

shapiro.test(sample(log(projections$Model_Score), 5000))
library(fitdistrplus)
descdist(projections$Model_Score, discrete=FALSE, boot=1000)

#PGA_Pred_list <- split(PGA_Prediction, PGA_Prediction$eventID)

library(fitdistrplus)

# 1. Descriptive statistics
descdist(projections$Model_Score)

# 2. Fit multiple distributions
candidates <- c("norm", "lnorm", "gamma", "weibull", "exp", "beta")
fits <- lapply(candidates, function(dist) {
  try(fitdist(projections$Model_Score, dist), silent=TRUE)
})

# 3. Compare valid fits
valid_fits <- fits[!sapply(fits, function(x) inherits(x, "try-error"))]
if(length(valid_fits) > 0) {
  comparison <- gofstat(valid_fits)
  print(comparison)
}

ks.test(projections$Model_Score, "punif", min=min(projections$Model_Score), max=max(projections$Model_Score))
# Check for ties/duplicates
length(unique(projections$Model_Score))  # vs length(projections$Model_Score)
table(projections$Model_Score)[1:10]     # See frequency of values

# Look at the distribution
hist(projections$Model_Score, breaks=30)
summary(projections$Model_Score)


library(fitdistrplus)

# Try distributions that handle right-skew well
fit_gamma <- fitdist(projections$Model_Score, "gamma")
fit_lnorm <- fitdist(projections$Model_Score, "lnorm")
fit_weibull <- fitdist(projections$Model_Score, "weibull")

# Compare fits
gofstat(list(fit_gamma, fit_lnorm, fit_weibull))
plot(fit_gamma)  # Check the best one


# Visual check of Weibull fit
plot(fit_weibull)

# Get parameters
fit_weibull$estimate
# Should show shape and scale parameters

# Overlay on histogram
hist(projections$Model_Score, freq=FALSE, breaks=30)
curve(dweibull(x, shape=fit_weibull$estimate[1], scale=fit_weibull$estimate[2]), 
      add=TRUE, col="red", lwd=2)
