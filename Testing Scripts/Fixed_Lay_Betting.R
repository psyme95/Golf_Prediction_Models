# Load packages
library(dplyr)

# Set working directory
set.seed(24)
setwd("C:/Projects/Golf/Results/")

# Load Data
all_predictions <- read.csv("./Win_RW_2207_1137/Model_Predictions.csv")

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

# Lay Betting
all_predictions_platt <- all_predictions %>%
  select("EventNumber", "EventID", "PlayerID", "Model_Score", "Actual_Position", "Win_odds", "Calibrated_Probability") %>%
  mutate(Lay_Odds = 1.1 * Win_odds,
         Model_Odds = 1 / Calibrated_Probability,
         Lay_Odds_Buffer = Lay_Odds + 0.0,
         Value_Bet = ifelse(Model_Odds > Lay_Odds_Buffer, 1, 0),
         Fixed_Stake = ifelse(Value_Bet == 1, 5, 0),
         FS_Liability = Fixed_Stake * (Lay_Odds - 1),
         FS_Outcome = ifelse(Actual_Position > 1, Fixed_Stake, -FS_Liability),
         Fixed_Liability = ifelse(Value_Bet == 1, 1000, 0),
         FL_Stake = Fixed_Liability / (Lay_Odds - 1),
         FL_Outcome = ifelse(Actual_Position > 1, FL_Stake, -Fixed_Liability))

# Calculate dead heat effects for Top 1 lay bets using official rules
all_predictions_platt_dh <- all_predictions_platt %>%
  # First, identify ties by grouping by EventID and Actual_Position
  group_by(EventID, Actual_Position) %>%
  mutate(
    # Count how many players tied at this position
    tied_players = n(),
    # Determine if this is a tie (more than 1 player at same position)
    is_tie = tied_players > 1
  ) %>%
  ungroup() %>%
  
  # For each tied group, determine if dead heat rules apply
  group_by(EventID, Actual_Position) %>%
  mutate(
    # Calculate the range of positions this tie occupies
    tie_start_position = Actual_Position,
    tie_end_position = Actual_Position + tied_players - 1,
    
    # Dead heat only applies if the tie crosses the Top 1 boundary (position 1)
    # i.e., some tied players finish ≤1 and some >1
    crosses_boundary = tie_start_position <= 1 & tie_end_position > 1,
    
    # For multi-winner markets, expected winners = remaining winning positions
    # If tie starts at position 4 and crosses to 6, there are 1-4+1 = 2 remaining positions
    # But only 1 of those (position 1) is still "winning" 
    DH_Expected_Winners = case_when(
      crosses_boundary ~ 1 - tie_start_position + 1,
      TRUE ~ NA_real_
    ),
    
    # Total number of tied players
    DH_Actual_Winners = case_when(
      crosses_boundary ~ tied_players,
      TRUE ~ NA_real_
    )
  ) %>%
  ungroup() %>%
  
  # Calculate dead heat returns and liability differences
  mutate(
    # Dead heat applies when:
    # 1. There's a tie that crosses the Top 1 boundary
    # 2. The player finished in Top 1 (lay bet loses)
    # 3. There's actually a liability to adjust
    loses_lay_bet = Actual_Position <= 1,
    has_dead_heat = crosses_boundary & loses_lay_bet & !is.na(DH_Expected_Winners),
    
    # Apply dead heat formula for 'Against' (lay) bets
    DH_Return = case_when(
      has_dead_heat & FS_Liability > 0 ~ 
        ((1 - (DH_Expected_Winners / DH_Actual_Winners)) * Lay_Odds) / 
        (Lay_Odds - 1) * FS_Liability,
      TRUE ~ 0
    ),
    
    # Also calculate for Fixed Liability bets if they exist
    DH_Return_FL = case_when(
      has_dead_heat & Fixed_Liability > 0 ~ 
        ((1 - (DH_Expected_Winners / DH_Actual_Winners)) * Lay_Odds) / 
        (Lay_Odds - 1) * Fixed_Liability,
      TRUE ~ 0
    ),
    
    # Calculate liability difference (savings from dead heat rules)
    DH_Liability_Difference = case_when(
      has_dead_heat & FS_Liability > 0 ~ FS_Liability - DH_Return,
      TRUE ~ 0
    ),
    
    DH_Liability_Difference_FL = case_when(
      has_dead_heat & Fixed_Liability > 0 ~ Fixed_Liability - DH_Return_FL,
      TRUE ~ 0
    ),
    
    # Create dead heat adjusted outcome columns
    # For Fixed Stake bets: Outcome = -Liability + Return
    FS_Outcome_DH = case_when(
      # If there's a dead heat adjustment, calculate net outcome
      has_dead_heat & FS_Liability > 0 ~ -FS_Liability + DH_Return,
      # Otherwise, use the original outcome
      TRUE ~ FS_Outcome
    ),
    
    # For Fixed Liability bets: Outcome = -Liability + Return  
    FL_Outcome_DH = case_when(
      # If there's a dead heat adjustment, calculate net outcome
      has_dead_heat & Fixed_Liability > 0 ~ -Fixed_Liability + DH_Return_FL,
      # Otherwise, use the original outcome
      TRUE ~ FL_Outcome
    )
  ) %>%
  
  # Remove temporary helper columns
  select(-tied_players, -is_tie, -tie_start_position, -tie_end_position, 
         -crosses_boundary, -loses_lay_bet, -has_dead_heat, -DH_Liability_Difference, 
         -DH_Liability_Difference_FL)

# Check total outcome
sum(all_predictions_platt$FS_Outcome)
sum(all_predictions_platt_dh$FS_Outcome_DH)
sum(all_predictions_platt_dh$FL_Outcome_DH)

# Comission
Event_Outcomes <- all_predictions_platt_dh %>%
  group_by(EventID) %>%
  summarise(Total_FS_Liability = sum(FS_Liability),
            Total_Outcome = sum(FS_Outcome_DH)) %>%
  mutate(Total_Outcome_Comission = case_when(
    Total_Outcome > 0 ~ Total_Outcome * 0.97,
    TRUE ~ Total_Outcome
  ))

write.csv(all_predictions_platt_dh, "./Lay_Betting/Win_Fixed.csv", row.names = F)