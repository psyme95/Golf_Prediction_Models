library(dplyr)
library(ggplot2)

# Calibrate predictions using logistic regression
logit_model_data <- data.frame(
  Model_Score = myBiomodEM@models.prediction@val$pred,
  Actual_Top20 = df_old$top_20,
  Top20_Odds = df_old$Top20_Odds
)

cal_model <- glm(Actual_Top20 ~ Model_Score + Top20_Odds, 
                         data = logit_model_data, 
                         family = binomial())


summary(cal_model)

new_preds <- all_predictions
new_probs <- predict(cal_model, 
        new_preds, 
        type = "response")

new_preds$Probability <- new_probs

# Function to calculate Kelly Criterion
calculate_kelly <- function(df) {
  df %>%
    mutate(
      # Convert odds to decimal format and calculate b (net odds)
      decimal_odds = Top20_Odds,
      b = decimal_odds - 1,
      
      # Use your model's probability
      p = Probability,
      q = 1 - p,
      
      # Calculate Kelly fraction
      kelly_fraction = (b * p - q) / b,
      
      # Set negative Kelly values to 0 (no bet)
      kelly_fraction = pmax(0, kelly_fraction),
      
      # Calculate recommended bet size as percentage of bankroll
      kelly_percentage = kelly_fraction * 100,
      
      # Alternative: Half-Kelly (more conservative)
      half_kelly_fraction = kelly_fraction * 0.5,
      half_kelly_percentage = half_kelly_fraction * 100
    )
}

# Apply Kelly Criterion to your data
new_preds_kelly <- calculate_kelly(new_preds)

# View results
head(new_preds_kelly %>% 
       select(EventID, PlayerID, Top20_Odds, Probability, kelly_fraction, 
              kelly_percentage, half_kelly_percentage))


# Additional analysis: Expected value
new_preds_kelly <- new_preds_kelly %>%
  mutate(
    expected_value = (Probability * (decimal_odds - 1)) - ((1 - Probability) * 1),
    positive_ev = expected_value > 0
  )

head(new_preds_kelly)

# Kelly Criterion Betting Simulation
# Function to simulate Kelly betting strategy
simulate_kelly_betting <- function(df, initial_bankroll = 1000, 
                                   strategy = "kelly", max_bet_amount = 10, max_bet_pct = 0.05,
                                   max_event_pct = 0.20) {
  
  # Sort by EventID to simulate chronological betting
  df_sorted <- df %>% arrange(EventID, PlayerID)
  
  # Get unique events in chronological order
  unique_events <- unique(df_sorted$EventID)
  
  # Initialize tracking variables
  bankroll <- initial_bankroll
  bet_history <- data.frame()
  
  # Track bankroll over time (after each event)
  bankroll_history <- data.frame(
    event_number = 0,
    EventID = NA,
    bankroll = initial_bankroll,
    cumulative_profit = 0,
    bets_in_event = 0
  )
  
  total_bet_counter <- 0
  
  # Process each event
  for(event_idx in 1:length(unique_events)) {
    current_event <- unique_events[event_idx]
    event_data <- df_sorted[df_sorted$EventID == current_event, ]
    
    # Store bankroll at start of event for bet sizing
    event_start_bankroll <- bankroll
    event_bets <- data.frame()
    bet_counter_for_event <- 0
    
    # First pass: Calculate all potential bets for this event
    potential_bets <- data.frame()
    
    for(i in 1:nrow(event_data)) {
      row <- event_data[i, ]
      
      # Ensure Kelly fractions are reasonable
      kelly_frac <- pmax(0, pmin(row$kelly_fraction, 1))
      half_kelly_frac <- kelly_frac * 0.5
      
      # Determine bet size based on strategy
      if(strategy == "kelly") {
        bet_fraction <- kelly_frac
      } else if(strategy == "half_kelly") {
        bet_fraction <- half_kelly_frac
      } else if(strategy == "fixed") {
        bet_fraction <- ifelse(kelly_frac > 0, 0.01, 0)
      }
      
      # Only consider if Kelly is positive and we have sufficient bankroll
      if(bet_fraction > 0 && event_start_bankroll > 10) {
        
        # Calculate theoretical bet amount (before event budget constraints)
        theoretical_bet <- 1000 * bet_fraction
        max_bankroll_bet <- 1000 * max_bet_pct
        individual_bet_amount <- min(theoretical_bet, max_bet_amount, max_bankroll_bet)
        # theoretical_bet <- 20
        # max_bankroll_bet <- 20
        # individual_bet_amount <- 20
        
        potential_bet <- data.frame(
          row_index = i,
          PlayerID = row$PlayerID,
          theoretical_bet = theoretical_bet,
          max_bankroll_bet = max_bankroll_bet,
          individual_bet_amount = individual_bet_amount,
          bet_fraction = bet_fraction,
          theoretical_fraction = bet_fraction,
          odds = row$decimal_odds,
          probability = row$Probability,
          kelly_fraction = row$kelly_fraction,
          expected_value = row$expected_value,
          bet_capped_amount = individual_bet_amount < theoretical_bet && theoretical_bet > max_bet_amount,
          bet_capped_pct = individual_bet_amount < theoretical_bet && theoretical_bet > max_bankroll_bet
        )
        
        potential_bets <- rbind(potential_bets, potential_bet)
      }
    }
    
    # Apply event budget constraint
    if(nrow(potential_bets) > 0) {
      total_event_allocation <- sum(potential_bets$individual_bet_amount)
      max_event_budget <- event_start_bankroll * max_event_pct
      
      # Check if we need to scale down due to event budget
      if(total_event_allocation > max_event_budget) {
        scaling_factor <- max_event_budget / total_event_allocation
        potential_bets$bet_amount <- potential_bets$individual_bet_amount * scaling_factor
        potential_bets$event_scaled <- TRUE
      } else {
        potential_bets$bet_amount <- potential_bets$individual_bet_amount
        potential_bets$event_scaled <- FALSE
      }
      
      # Now place the actual bets
      for(j in 1:nrow(potential_bets)) {
        bet_counter_for_event <- bet_counter_for_event + 1
        total_bet_counter <- total_bet_counter + 1
        
        bet_data <- potential_bets[j, ]
        original_row <- event_data[bet_data$row_index, ]
        
        # Record this bet (outcome determined later)
        bet_record <- data.frame(
          bet_number = total_bet_counter,
          event_number = event_idx,
          EventID = original_row$EventID,
          PlayerID = bet_data$PlayerID,
          theoretical_bet = bet_data$theoretical_bet,
          max_bankroll_bet = bet_data$max_bankroll_bet,
          individual_bet_amount = bet_data$individual_bet_amount,
          bet_amount = bet_data$bet_amount,
          bet_fraction = bet_data$bet_amount / event_start_bankroll,
          theoretical_fraction = bet_data$theoretical_fraction,
          odds = bet_data$odds,
          probability = bet_data$probability,
          kelly_fraction = bet_data$kelly_fraction,
          expected_value = bet_data$expected_value,
          bet_capped_amount = bet_data$bet_capped_amount,
          bet_capped_pct = bet_data$bet_capped_pct,
          bet_capped = bet_data$bet_amount < bet_data$theoretical_bet,
          event_scaled = bet_data$event_scaled,
          total_event_allocation = total_event_allocation,
          max_event_budget = max_event_budget,
          # Outcome fields (filled after event)
          won = NA,
          #profit = NA,
          bankroll_after = NA
        )
        
        event_bets <- rbind(event_bets, bet_record)
      }
    }
    
    # Now resolve all bets for this event
    if(nrow(event_bets) > 0) {
      total_event_profit <- 0
      
      for(j in 1:nrow(event_bets)) {
        bet <- event_bets[j, ]
        player_result <- event_data[event_data$PlayerID == bet$PlayerID, ]
        
        # Determine outcome
        won_bet <- player_result$Actual_Top20 == 1
        
        if(won_bet) {
          profit <- (bet$bet_amount / 10) * player_result$Top20_Profit
          } else {
          profit <- -bet$bet_amount
        }
        
        total_event_profit <- total_event_profit + profit
        
        # Update bet record
        event_bets[j, "won"] <- won_bet
        event_bets[j, "profit"] <- profit
      }
      
      # Update bankroll after all bets in event are resolved
      bankroll <- bankroll + total_event_profit
      
      # Update bankroll_after for all bets in this event
      event_bets$bankroll_after <- bankroll
      
      # Add to overall bet history
      bet_history <- rbind(bet_history, event_bets)
    }
    
    # Record bankroll after this event
    bankroll_record <- data.frame(
      event_number = event_idx,
      EventID = current_event,
      bankroll = bankroll,
      cumulative_profit = bankroll - initial_bankroll,
      bets_in_event = bet_counter_for_event
    )
    
    bankroll_history <- rbind(bankroll_history, bankroll_record)
  }
  
  return(list(
    bet_history = bet_history,
    bankroll_history = bankroll_history,
    final_bankroll = bankroll,
    total_profit = bankroll - initial_bankroll,
    total_bets = total_bet_counter
  ))
}

# Run simulations for different strategies with event budget limits
kelly_sim <- simulate_kelly_betting(new_preds_kelly, strategy = "kelly", max_bet_amount = 10, max_bet_pct = 0.01, max_event_pct = 0.5)
half_kelly_sim <- simulate_kelly_betting(new_preds_kelly, strategy = "half_kelly", max_bet_amount = 10, max_bet_pct = 0.01, max_event_pct = 0.5)
fixed_sim <- simulate_kelly_betting(new_preds_kelly, strategy = "fixed", max_bet_amount = 10, max_bet_pct = 0.01, max_event_pct = 0.5)

# Print summary results
print_simulation_summary <- function(sim_result, strategy_name) {
  cat(paste("\n", strategy_name, "Strategy Results:\n"))
  cat("==================================\n")
  cat("Initial Bankroll: £1,000\n")
  cat("Final Bankroll: £", round(sim_result$final_bankroll, 2), "\n")
  cat("Total Profit: £", round(sim_result$total_profit, 2), "\n")
  cat("ROI: ", round((sim_result$total_profit / 1000) * 100, 2), "%\n")
  cat("Total Bets Placed:", sim_result$total_bets, "\n")
  
  if(sim_result$total_bets > 0) {
    win_rate <- mean(sim_result$bet_history$won) * 100
    avg_bet_size <- mean(sim_result$bet_history$bet_amount)
    cat("Win Rate: ", round(win_rate, 2), "%\n")
    cat("Average Bet Size: £", round(avg_bet_size, 2), "\n")
    cat("Largest Bet: £", round(max(sim_result$bet_history$bet_amount), 2), "\n")
    cat("Biggest Win: £", round(max(sim_result$bet_history$profit), 2), "\n")
    cat("Biggest Loss: £", round(min(sim_result$bet_history$profit), 2), "\n")
    
    # Show how often bets were capped
    if("bet_capped" %in% names(sim_result$bet_history)) {
      capped_bets <- sum(sim_result$bet_history$bet_capped)
      capped_by_amount <- sum(sim_result$bet_history$bet_capped_amount, na.rm = TRUE)
      capped_by_pct <- sum(sim_result$bet_history$bet_capped_pct, na.rm = TRUE)
      event_scaled_bets <- sum(sim_result$bet_history$event_scaled, na.rm = TRUE)
      
      cat("Total bets capped:", capped_bets, "out of", sim_result$total_bets, 
          paste0("(", round(capped_bets/sim_result$total_bets * 100, 1), "%)\n"))
      cat("  - Capped by amount limit:", capped_by_amount, "\n")
      cat("  - Capped by percentage limit:", capped_by_pct, "\n")
      cat("  - Scaled by event budget:", event_scaled_bets, "\n")
    }
  }
}

print_simulation_summary(kelly_sim, "Full Kelly")
print_simulation_summary(half_kelly_sim, "Half Kelly")
print_simulation_summary(fixed_sim, "Fixed 2%")

# Create visualization comparing strategies
if(nrow(kelly_sim$bankroll_history) > 0 && 
   nrow(half_kelly_sim$bankroll_history) > 0 && 
   nrow(fixed_sim$bankroll_history) > 0) {
  
  # Combine bankroll histories
  combined_history <- rbind(
    kelly_sim$bankroll_history %>% mutate(strategy = "Full Kelly"),
    half_kelly_sim$bankroll_history %>% mutate(strategy = "Half Kelly"),
    fixed_sim$bankroll_history %>% mutate(strategy = "Fixed 2%")
  )
  
  # Plot bankroll over time
  p1 <- ggplot(combined_history, aes(x = event_number, y = bankroll, color = strategy)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 1000, linetype = "dashed", alpha = 0.5) +
    labs(title = "Bankroll Growth Over Time",
         x = "Event Number", y = "Bankroll (£)",
         color = "Strategy") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  print(p1)
}

write.csv(kelly_sim[["bet_history"]], "./Betting Simulations/BH_Logit_Kelly_Simulation.csv", row.names=F)
write.csv(kelly_sim[["bankroll_history"]], "./Betting Simulations/Bankroll_Logit_Kelly_Simulation.csv", row.names=F)
