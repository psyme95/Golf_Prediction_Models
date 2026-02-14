# ===== ROUND 2 CALIBRATION MODELING WITH TRAIN/TEST SPLIT =====
# This script splits data, trains calibration models, and evaluates performance

# ===== CONFIGURATION =====
set.seed(42)

# Train/test split configuration
TRAIN_PERCENTAGE <- 0.70  # 70% for training, 30% for testing
SPLIT_BY_EVENT <- TRUE     # Split by event (recommended) vs random split

# Define markets to calibrate
MARKETS <- list(
  "win" = list(target = "win", label = "Winner", market_size = 1),
  "top5" = list(target = "top5", label = "Top 5", market_size = 5),
  "top10" = list(target = "top10", label = "Top 10", market_size = 10),
  "top20" = list(target = "top20", label = "Top 20", market_size = 20),
  "top40" = list(target = "top40", label = "Top 40", market_size = 40)
)

# ===== LOAD PACKAGES =====
library(dplyr)
library(ggplot2)
library(pROC)

# ===== SPLIT DATA INTO TRAIN/TEST =====
split_train_test <- function(data, train_pct = 0.70, by_event = TRUE) {
  
  cat("\n=== SPLITTING DATA ===\n")
  cat("Total records:", nrow(data), "\n")
  cat("Unique events:", length(unique(data$eventID)), "\n")
  cat("Split method:", ifelse(by_event, "By Event", "Random"), "\n")
  
  if (by_event) {
    # Split by event to prevent data leakage
    unique_events <- unique(data$eventID)
    n_train_events <- floor(length(unique_events) * train_pct)
    
    # Randomly select training events
    train_events <- sample(unique_events, n_train_events)
    
    train_data <- data %>% filter(eventID %in% train_events)
    test_data <- data %>% filter(!eventID %in% train_events)
    
    cat("Training events:", n_train_events, "\n")
    cat("Testing events:", length(unique_events) - n_train_events, "\n")
    
  } else {
    # Random split
    train_indices <- sample(1:nrow(data), floor(nrow(data) * train_pct))
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
  }
  
  cat("Training records:", nrow(train_data), 
      "(", round(100 * nrow(train_data) / nrow(data), 1), "%)\n")
  cat("Testing records:", nrow(test_data), 
      "(", round(100 * nrow(test_data) / nrow(data), 1), "%)\n")
  
  return(list(train = train_data, test = test_data))
}

# ===== CALIBRATION FUNCTIONS =====

# Platt Scaling implementation
apply_platt_scaling <- function(scores, labels) {
  sigmoid <- function(x, A, B) {
    1 / (1 + exp(A * x + B))
  }
  
  neg_log_likelihood <- function(params) {
    A <- params[1]
    B <- params[2]
    probs <- sigmoid(scores, A, B)
    probs <- pmax(pmin(probs, 1 - 1e-15), 1e-15)
    -sum(labels * log(probs) + (1 - labels) * log(1 - probs))
  }
  
  result <- optim(c(0, 0), neg_log_likelihood, method = "BFGS")
  return(list(A = result$par[1], B = result$par[2]))
}

predict_platt_scaling <- function(scores, platt_params) {
  1 / (1 + exp(platt_params$A * scores + platt_params$B))
}

# ===== BUILD CALIBRATION MODELS =====

build_rd2_calibration_models <- function(train_data, market_config) {
  
  market_name <- market_config$label
  target_var <- market_config$target
  
  # Prepare data - remove any missing values
  model_data <- train_data %>%
    filter(!is.na(Model_Score_Median) & !is.na(Rd2Pos) & !is.na(Rd2Lead)) %>%
    filter(!is.na(!!sym(target_var)))
  
  # Model 1: GLM with all three predictors
  cat("Fitting GLM with all predictors...\n")
  glm_full <- glm(
    as.formula(paste(target_var, "~ Model_Score_Median + Rd2Pos + Rd2Lead")),
    data = model_data,
    family = binomial()
  )
  
  # Model 2: GLM with Model_Score and Rd2Pos only
  cat("Fitting GLM with Model Score + Rd2 Position...\n")
  glm_score_pos <- glm(
    as.formula(paste(target_var, "~ Model_Score_Median + Rd2Pos")),
    data = model_data,
    family = binomial()
  )
  
  # Model 3: GLM with interactions
  cat("Fitting GLM with interactions...\n")
  glm_interact <- glm(
    as.formula(paste(target_var, "~ Model_Score_Median * Rd2Pos + Rd2Lead")),
    data = model_data,
    family = binomial()
  )
  
  # Model 4: Platt Scaling on Model_Score only (baseline)
  cat("Fitting Platt Scaling (baseline)...\n")
  platt_params <- apply_platt_scaling(
    model_data$Model_Score_Median, 
    model_data[[target_var]]
  )
  
  # Model 5: GLM with polynomial terms
  cat("Fitting GLM with polynomial terms...\n")
  glm_poly <- glm(
    as.formula(paste(target_var, "~ poly(Model_Score_Median, 2) + poly(Rd2Pos, 2) + Rd2Lead")),
    data = model_data,
    family = binomial()
  )
  
  # Return all models
  return(list(
    glm_full = glm_full,
    glm_score_pos = glm_score_pos,
    glm_interact = glm_interact,
    glm_poly = glm_poly,
    platt_params = platt_params
  ))
}

# ===== GENERATE PREDICTIONS =====
generate_predictions <- function(data, calibration_models, market_config) {
  
  target_var <- market_config$target
  
  # Prepare prediction data
  pred_data <- data %>%
    filter(!is.na(Model_Score_Median) & !is.na(Rd2Pos) & !is.na(Rd2Lead))
  
  # Generate predictions from all models
  pred_data[[paste0(target_var, "_GLM_Full")]] <- predict(
    calibration_models$glm_full, 
    newdata = pred_data, 
    type = "response"
  )
  
  pred_data[[paste0(target_var, "_GLM_Score_Pos")]] <- predict(
    calibration_models$glm_score_pos, 
    newdata = pred_data, 
    type = "response"
  )
  
  pred_data[[paste0(target_var, "_GLM_Interact")]] <- predict(
    calibration_models$glm_interact, 
    newdata = pred_data, 
    type = "response"
  )
  
  pred_data[[paste0(target_var, "_GLM_Poly")]] <- predict(
    calibration_models$glm_poly, 
    newdata = pred_data, 
    type = "response"
  )
  
  pred_data[[paste0(target_var, "_Platt")]] <- predict_platt_scaling(
    pred_data$Model_Score_Median, 
    calibration_models$platt_params
  )
  
  # Calculate ensemble (average of GLM methods, excluding baseline Platt)
  pred_data[[paste0(target_var, "_Ensemble")]] <- rowMeans(
    pred_data[, c(paste0(target_var, "_GLM_Full"),
                  paste0(target_var, "_GLM_Score_Pos"),
                  paste0(target_var, "_GLM_Interact"),
                  paste0(target_var, "_GLM_Poly"))]
  )
  
  return(pred_data)
}

# ===== EVALUATE MODEL PERFORMANCE =====
evaluate_model_performance <- function(predictions, market_config) {
  
  target_var <- market_config$target
  actual <- predictions[[target_var]]
  
  # Define prediction columns
  pred_cols <- c(
    paste0(target_var, "_GLM_Full"),
    paste0(target_var, "_GLM_Score_Pos"),
    paste0(target_var, "_GLM_Interact"),
    paste0(target_var, "_GLM_Poly"),
    paste0(target_var, "_Platt"),
    paste0(target_var, "_Ensemble")
  )
  
  # Calculate metrics for each model
  results <- data.frame()
  
  for (pred_col in pred_cols) {
    predicted <- predictions[[pred_col]]
    
    # Log Loss
    predicted_safe <- pmax(pmin(predicted, 1 - 1e-15), 1e-15)
    log_loss <- -mean(actual * log(predicted_safe) + (1 - actual) * log(1 - predicted_safe))
    
    # Brier Score
    brier_score <- mean((predicted - actual)^2)
    
    # AUC
    roc_obj <- roc(actual, predicted, quiet = TRUE)
    auc_score <- as.numeric(auc(roc_obj))
    
    # Calibration metrics (expected vs observed)
    # Use unique quantiles to avoid duplicate breaks
    calibration_error <- tryCatch({
      # Get unique quantile breaks
      breaks <- unique(quantile(predicted, probs = seq(0, 1, 0.1)))
      
      # Only calculate calibration if we have enough unique breaks
      if (length(breaks) > 2) {
        pred_bins <- cut(predicted, breaks = breaks, include.lowest = TRUE, labels = FALSE)
        
        calibration <- predictions %>%
          mutate(bin = pred_bins, 
                 actual = actual,
                 predicted = predicted) %>%
          filter(!is.na(bin)) %>%
          group_by(bin) %>%
          summarise(
            expected = mean(predicted),
            observed = mean(actual),
            count = n(),
            .groups = "drop"
          )
        
        mean(abs(calibration$expected - calibration$observed))
      } else {
        # Not enough unique values for binning
        NA
      }
    }, error = function(e) {
      # Return NA if calibration calculation fails
      NA
    })
    
    # Store results
    model_name <- gsub(paste0(target_var, "_"), "", pred_col)
    results <- rbind(results, data.frame(
      Model = model_name,
      LogLoss = log_loss,
      BrierScore = brier_score,
      AUC = auc_score,
      CalibrationError = calibration_error
    ))
  }
  
  # Sort by Log Loss (lower is better)
  results <- results %>% arrange(LogLoss)
  
  return(results)
}

# ===== MAIN EXECUTION =====
# Step 1: Split data
split_data <- split_train_test(pga, TRAIN_PERCENTAGE, SPLIT_BY_EVENT)
train_data <- split_data$train
test_data <- split_data$test

# Step 2: Train models for all markets
all_calibration_models <- list()
all_test_predictions <- list()
all_performance <- list()

for (market_name in names(MARKETS)) {
  market_config <- MARKETS[[market_name]]
  
  # Build models on training data
  calibration_models <- build_rd2_calibration_models(train_data, market_config)
  all_calibration_models[[market_name]] <- calibration_models
  
  # Generate predictions on test data
  test_predictions <- generate_predictions(test_data, calibration_models, market_config)
  all_test_predictions[[market_name]] <- test_predictions
  
  # Evaluate performance
  performance <- evaluate_model_performance(test_predictions, market_config)
  all_performance[[market_name]] <- performance
  
  cat("\n--- Test Performance for", market_config$label, "---\n")
  print(performance, row.names = FALSE)
}

# ===== CREATE NORMALIZED PROBABILITIES =====
# For each market, create normalized probabilities using the ensemble predictions
normalized_predictions <- test_data %>%
  select(eventID, playerID, posn, Rd2Pos, Rd2Lead, Model_Score_Median)

for (market_name in names(MARKETS)) {
  market_config <- MARKETS[[market_name]]
  target_var <- market_config$target
  market_size <- market_config$market_size
  
  # Get ensemble predictions
  predictions <- all_test_predictions[[market_name]]
  ensemble_col <- paste0(target_var, "_Ensemble")
  
  # Add to normalized predictions dataframe
  normalized_predictions[[paste0(target_var, "_Prob")]] <- predictions[[ensemble_col]]
  
  # Calculate normalized probabilities by event
  normalized_predictions <- normalized_predictions %>%
    group_by(eventID) %>%
    mutate(
      !!paste0(target_var, "_Prob_Norm") := 
        (!!sym(paste0(target_var, "_Prob")) / sum(!!sym(paste0(target_var, "_Prob")))) * market_size,
      !!paste0(target_var, "_Odds") := 1 / !!sym(paste0(target_var, "_Prob")),
      !!paste0(target_var, "_Odds_Norm") := 1 / !!sym(paste0(target_var, "_Prob_Norm"))
    ) %>%
    ungroup()
}


# ===== EXPORT RESULTS =====
# Save test predictions with all model probabilities
results_file <- paste0("Round2_Test_Predictions_", format(Sys.time(), "%Y%m%d_%H%M"), ".csv")
write.csv(normalized_predictions, results_file, row.names = FALSE)
