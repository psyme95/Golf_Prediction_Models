library(biomod2)
library(tidyverse)
library(readxl)
library(openxlsx)

setwd("C:/Projects/Golf/Weekly_Modelling")

# ============================================
# CONFIG
# ============================================
CURRENT_SEASON <- "S26"
PREDICTION_DATE <- format(Sys.Date(), "%d-%m-%Y")

# ============================================
# LOAD MODEL
# ============================================
load_model <- function(tour_name, season = CURRENT_SEASON) {
  path <- file.path("./Output/Models", paste0(tour_name, "_R2_", season, "_trained.rds"))
  if (!file.exists(path)) stop("Model not found: ", path)
  cat("Loading model:", path, "\n")
  model <- readRDS(path)
  cat("  Trained:", as.character(model$trained_at), "\n")
  cat("  Seasons:", paste(model$train_seasons, collapse = ", "), "\n")
  cat("  Runs:", model$n_runs, "\n")
  model
}

# ============================================
# PREPARE DATA
# ============================================
prepare_pga_data <- function(prediction_date = PREDICTION_DATE) {
  
  cat("\n--- Preparing PGA data ---\n")
  
  cat("Reading This_Week_Rd2_PGA.csv... ")
  pga_preds <- read.csv("./Input/This_Week_Rd2_PGA.csv")
  cat(nrow(pga_preds), "rows\n")

  pga_preds <- pga_preds %>%
    rename(
      GLM_Odds_Probability_Median = Normalised_Probability,
      Model_Score_Median = Model_Score
    )
  
  missing <- pga_preds %>% filter(is.na(Top5) | is.na(Rd2Pos) | is.na(AvPosn) | is.na(Betfair_rd2))
  if (nrow(missing) > 0) {
    cat("\nWARNING:", nrow(missing), "players with missing data:\n")
    print(missing %>% select(Surname, Firstname, Top5, AvPosn, Rd2Pos, Betfair_rd2))
  }
  
  result <- pga_preds %>% drop_na()
  cat("\nFinal dataset:", nrow(result), "players\n")
  
  result
}

prepare_euro_data <- function(prediction_date = PREDICTION_DATE) {
  
  cat("\n--- Preparing Euro data ---\n")
  
  cat("Reading This_Week_Rd2_Euro.csv... ")
  pga_preds <- read.csv("./Input/This_Week_Rd2_Euro.csv")
  cat(nrow(pga_preds), "rows\n")
  
  pga_preds <- pga_preds %>%
    rename(
      GLM_Odds_Probability_Median = Normalised_Probability,
      Model_Score_Median = Model_Score
    )
  
  missing <- pga_preds %>% filter(is.na(Top5) | is.na(Rd2Pos) | is.na(AvPosn) | is.na(Betfair_rd2))
  if (nrow(missing) > 0) {
    cat("\nWARNING:", nrow(missing), "players with missing data:\n")
    print(missing %>% select(Surname, Firstname, Top5, AvPosn, Rd2Pos, Betfair_rd2))
  }
  
  result <- pga_preds %>% drop_na()
  cat("\nFinal dataset:", nrow(result), "players\n")
  
  result
}

# ============================================
# PREDICT
# ============================================
predict_r2 <- function(new_data, model_package) {
  
  cat("\n--- Generating predictions ---\n")
  cat("Tour:", model_package$tour_name, "\n")
  cat("Input rows:", nrow(new_data), "\n")
  
  model_vars <- model_package$model_vars
  n_runs <- model_package$n_runs
  
  cat("Checking required variables... ")
  missing <- setdiff(model_vars, names(new_data))
  if (length(missing) > 0) stop("Missing variables: ", paste(missing, collapse = ", "))
  if (!"Betfair_rd2" %in% names(new_data)) stop("Missing Betfair_rd2 for calibration")
  cat("OK\n")
  
  expl <- new_data %>% select(all_of(model_vars)) %>% as.data.frame()
  coords <- data.frame(x = seq_len(nrow(new_data)), y = seq_len(nrow(new_data)))
  
  all_scores <- matrix(NA, nrow = nrow(new_data), ncol = n_runs)
  all_calibrated <- matrix(NA, nrow = nrow(new_data), ncol = n_runs)
  
  for (run in seq_len(n_runs)) {
    
    cat("  Run", run, "of", n_runs, "... ")
    
    run_result <- model_package$run_results[[run]]
    
    proj <- BIOMOD_EnsembleForecasting(
      bm.em = run_result$ensemble_model,
      bm.proj = NULL,
      proj.name = paste0("Pred_Run", run),
      new.env = expl,
      new.env.xy = coords,
      models.chosen = "all"
    )
    
    scores <- proj@proj.out@val$pred
    all_scores[, run] <- scores
    
    cal_data <- data.frame(
      oof_scores = scores,
      odds = new_data$Betfair_rd2
    )
    
    all_calibrated[, run] <- predict(
      run_result$calibration_model,
      newdata = cal_data,
      type = "response"
    )
    
    cat("done (score range:", round(min(scores), 1), "-", round(max(scores), 1), ")\n")
  }
  
  cat("Averaging across runs and calculating probabilities... ")
  
  result <- new_data %>%
    mutate(
      Rd2_Model_Score = rowMeans(all_scores),
      Rd2_Calibrated_Prob = rowMeans(all_calibrated),
      Rd2_Model_Odds = 1 / Rd2_Calibrated_Prob
    ) %>%
    mutate(
      Rd2_Normalised_Prob = Rd2_Calibrated_Prob / sum(Rd2_Calibrated_Prob),
      Rd2_Normalised_Odds = 1 / Rd2_Normalised_Prob
    )
  
  cat("done\n")
  
  cat("\nPrediction summary:\n")
  cat("  Calibrated prob range:", round(min(result$Rd2_Calibrated_Prob), 4), "-", 
      round(max(result$Rd2_Calibrated_Prob), 4), "\n")
  cat("  Model odds range:", round(min(result$Rd2_Model_Odds), 1), "-", 
      round(max(result$Rd2_Model_Odds), 1), "\n")
  
  result
}

# ============================================
# MAIN
# ============================================

cat("=== R2 Prediction Script ===\n")
cat("Date:", PREDICTION_DATE, "\n")
cat("Season:", CURRENT_SEASON, "\n")

# Load models
pga_model <- load_model("PGA")
euro_model <- load_model("Euro")

# Prepare data
pga_data <- prepare_pga_data()
euro_data <- prepare_euro_data()

# Generate predictions
pga_predictions <- predict_r2(pga_data, pga_model)
euro_predictions <- predict_r2(euro_data, euro_model)

# Export
pga_out <- paste0("./Output/Predictions/PGA_R2_Predictions_", PREDICTION_DATE, ".xlsx")
euro_out <- paste0("./Output/Predictions/Euro_R2_Predictions_", PREDICTION_DATE, ".xlsx")

write.xlsx(pga_predictions, pga_out)
write.xlsx(euro_predictions, euro_out)

cat("\n=== Complete ===\n")
cat("Exported:\n")
cat(" ", pga_out, "\n")
cat(" ", euro_out, "\n")