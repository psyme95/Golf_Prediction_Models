# Load required libraries
library(dplyr)
library(ggplot2)
library(caret)

# ===== MODEL CALIBRATION =====
# Platt Scaling
apply_platt_scaling <- function(scores, labels) {
  # Fit sigmoid parameters A and B using maximum likelihood
  sigmoid <- function(x, A, B) {
    1 / (1 + exp(A * x + B))
  }
  
  # Objective function to minimize (negative log-likelihood)
  neg_log_likelihood <- function(params) {
    A <- params[1]
    B <- params[2]
    probs <- sigmoid(scores, A, B)
    # Add small epsilon to avoid log(0)
    probs <- pmax(pmin(probs, 1 - 1e-15), 1e-15)
    -sum(labels * log(probs) + (1 - labels) * log(1 - probs))
  }
  
  # Optimize parameters
  result <- optim(c(0, 0), neg_log_likelihood, method = "BFGS")
  
  return(list(A = result$par[1], B = result$par[2]))
}

apply_platt_scaling_caret <- function(df, score_col = "Model_Score", label_col = "Actual_Win") {
  # Create a simple model wrapper for your scores
  scores <- df[[score_col]]
  labels <- factor(df[[label_col]], levels = c(0, 1))
  
  # Create folds for cross-validation
  folds <- createFolds(labels, k = 5, returnTrain = TRUE)
  
  platt_probs <- numeric(length(scores))
  
  for(i in 1:length(folds)) {
    train_idx <- folds[[i]]
    val_idx <- setdiff(1:length(scores), train_idx)
    
    # Fit Platt scaling on fold
    platt_params <- apply_platt_scaling(scores[train_idx], as.numeric(labels[train_idx]) - 1)
    
    # Apply to validation set
    platt_probs[val_idx] <- 1 / (1 + exp(platt_params$A * scores[val_idx] + platt_params$B))
  }
  
  df$Platt_Probability <- platt_probs
  return(df)
}

all_predictions_platt <- apply_platt_scaling_caret(all_predictions)
all_predictions_platt$Calibrated_Probability <- all_predictions_platt$Platt_Probability

all_predictions_platt <- all_predictions_platt %>%
  mutate(Prob_Edge = Calibrated_Probability - Implied_Probability,
         Expected_Value = (Calibrated_Probability * (Win_odds - 1)) - (1 - Calibrated_Probability),
         Kelly_Fraction = pmax(0, 
                               (Calibrated_Probability * Win_odds - 1) / (Win_odds - 1))) %>%
  select(-Platt_Probability)


# Calibration plot and statistics
cal_data <- all_predictions %>%
  mutate(
    prob_bin = cut(Calibrated_Probability, 
                   breaks = seq(0, 1, 0.01), 
                   include.lowest = TRUE)
                   #labels = paste0(seq(0, 90, 10), "-", seq(10, 100, 10), "%"))
  ) %>%
  group_by(prob_bin) %>%
  summarise(
    count = n(),
    predicted_prob = mean(Calibrated_Probability),
    actual_rate = mean(Actual_Win),
    difference = predicted_prob - actual_rate,
    .groups = 'drop'
  ) %>%
  filter(count >= 5)  # Only bins with sufficient data


# Calibration plot
cal_plot <- ggplot(cal_data, aes(x = predicted_prob, y = actual_rate)) +
  geom_point(aes(size = count), alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  labs(x = "Predicted Probability",
       y = "Actual Top 20 Rate",
       size = "Number of Bets") +
  theme_minimal()
  
plot(cal_plot)
