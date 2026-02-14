library(dplyr)
library(ggplot2)

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
    
    )
}

# Apply Kelly Criterion to your data
new_preds_kelly <- calculate_kelly(new_preds)

# Additional analysis: Expected value
new_preds_kelly <- new_preds_kelly %>%
  mutate(
    expected_value = (Probability * (decimal_odds - 1)) - ((1 - Probability) * 1),
    positive_ev = expected_value > 0
  )

head(new_preds_kelly)

# Function for Kelly simulation
simulate_kelly_betting_custom <- function(df, 
                                          initial_bankroll = 1000,
                                          kelly_multiplier = 0.5,        # Fraction of Kelly to bet (0.5 = half Kelly)
                                          max_bet_amount = 50,           # Maximum bet in currency units
                                          max_bet_pct = 0.05,            # Maximum bet as % of bankroll
                                          max_event_pct = 0.20,          # Maximum total bets per event as % of bankroll
                                          min_kelly_threshold = 0.02,    # Minimum Kelly fraction to consider betting
                                          min_expected_value = 0.05,     # Minimum expected value to bet
                                          min_prob_edge = 0.03,          # Minimum probability edge over market
                                          max_bets_per_event = 5,        # Maximum number of bets per event
                                          use_dynamic_bankroll = FALSE,   # Whether to use current bankroll for bet sizing
                                          min_odds = 1,
                                          max_odds = 1000) { 
  
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
    bankroll_start = initial_bankroll,
    bankroll_end = initial_bankroll,
    event_profit = 0,
    cumulative_profit = 0,
    bets_in_event = 0,
    total_bet_amount = 0
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
    
    # Calculate potential bets for this event
    potential_bets <- data.frame()
    
    for(i in 1:nrow(event_data)) {
      row <- event_data[i, ]
      
      # Calculate probability edge over market
      implied_prob <- 1 / row$Top20_Odds
      prob_edge <- row$Probability - implied_prob
      
      # Ensure Kelly fraction is reasonable (cap at 100%)
      kelly_frac <- pmax(0, pmin(row$kelly_fraction, 1))
      
      # Apply Kelly multiplier (e.g., 0.25 for quarter Kelly)
      adjusted_kelly <- kelly_frac * kelly_multiplier
      
      # Calculate reference bankroll once
      reference_bankroll <- ifelse(use_dynamic_bankroll, event_start_bankroll, initial_bankroll)
      
      # Apply all filters - must pass ALL conditions
      passes_filters <- (
        adjusted_kelly > min_kelly_threshold &&
          row$expected_value > min_expected_value &&
          prob_edge > min_prob_edge &&
          reference_bankroll > max_bet_amount &&
          row$Top20_Odds >= min_odds &&
          row$Top20_Odds <= max_odds

      )
      
      if(passes_filters) {
        # Calculate theoretical bet amount
        theoretical_bet <- reference_bankroll * adjusted_kelly
        
        # Apply individual bet constraints
        max_bankroll_bet <- reference_bankroll * max_bet_pct
        constrained_bet <- min(theoretical_bet, max_bet_amount, max_bankroll_bet)
        
        # Store potential bet
        potential_bet <- data.frame(
          row_index = i,
          PlayerID = row$PlayerID,
          theoretical_bet = theoretical_bet,
          constrained_bet = constrained_bet,
          adjusted_kelly = adjusted_kelly,
          original_kelly = kelly_frac,
          kelly_multiplier_used = kelly_multiplier,
          odds = row$decimal_odds,
          probability = row$Probability,
          expected_value = row$expected_value,
          prob_edge = prob_edge,
          implied_prob = implied_prob,
          bet_capped_amount = constrained_bet < theoretical_bet && theoretical_bet > max_bet_amount,
          bet_capped_pct = constrained_bet < theoretical_bet && theoretical_bet > max_bankroll_bet,
          stringsAsFactors = FALSE
        )
        
        potential_bets <- rbind(potential_bets, potential_bet)
      }
    }
    
    # Apply maximum bets per event filter (keep best bets)
    if(nrow(potential_bets) > max_bets_per_event) {
      potential_bets <- potential_bets %>%
        arrange(desc(expected_value)) %>%
        slice_head(n = max_bets_per_event)
    }
    
    # Apply event budget constraint
    if(nrow(potential_bets) > 0) {
      total_event_allocation <- sum(potential_bets$constrained_bet)
      max_event_budget <- reference_bankroll * max_event_pct
      
      # Scale down if exceeding event budget
      if(total_event_allocation > max_event_budget) {
        scaling_factor <- max_event_budget / total_event_allocation
        potential_bets$final_bet_amount <- potential_bets$constrained_bet * scaling_factor
        potential_bets$event_scaled <- TRUE
      } else {
        potential_bets$final_bet_amount <- potential_bets$constrained_bet
        potential_bets$event_scaled <- FALSE
      }
      
      # Place the actual bets and record them
      event_total_bet <- 0
      
      for(j in 1:nrow(potential_bets)) {
        bet_counter_for_event <- bet_counter_for_event + 1
        total_bet_counter <- total_bet_counter + 1
        
        bet_data <- potential_bets[j, ]
        original_row <- event_data[bet_data$row_index, ]
        
        event_total_bet <- event_total_bet + bet_data$final_bet_amount
        
        # Record this bet
        bet_record <- data.frame(
          bet_number = total_bet_counter,
          event_number = event_idx,
          EventID = original_row$EventID,
          PlayerID = bet_data$PlayerID,
          bet_amount = bet_data$final_bet_amount,
          theoretical_bet = bet_data$theoretical_bet,
          constrained_bet = bet_data$constrained_bet,
          final_bet_fraction = bet_data$final_bet_amount / event_start_bankroll,
          adjusted_kelly = bet_data$adjusted_kelly,
          original_kelly = bet_data$original_kelly,
          kelly_multiplier = bet_data$kelly_multiplier_used,
          odds = bet_data$odds,
          probability = bet_data$probability,
          expected_value = bet_data$expected_value,
          prob_edge = bet_data$prob_edge,
          implied_prob = bet_data$implied_prob,
          bet_capped_amount = bet_data$bet_capped_amount,
          bet_capped_pct = bet_data$bet_capped_pct,
          event_scaled = bet_data$event_scaled,
          bankroll_at_bet = event_start_bankroll,
          # Outcome fields (filled after event resolution)
          won = NA,
          profit = NA,
          stringsAsFactors = FALSE
        )
        
        event_bets <- rbind(event_bets, bet_record)
      }
      
      # Resolve all bets for this event
      total_event_profit <- 0
      
      for(j in 1:nrow(event_bets)) {
        bet <- event_bets[j, ]
        player_result <- event_data[event_data$PlayerID == bet$PlayerID, ]
        
        # Determine outcome and calculate profit
        won_bet <- player_result$Actual_Top20 == 1
        
        if(won_bet) {
          # Win: Determine winning based on TopX_Profit column - Profit column based on £10 stake so get the multiplier for the profit
          profit <- (bet$bet_amount / 10) * player_result$Top20_Profit
        } else {
          # Loss: lose bet amount
          profit <- -bet$bet_amount
        }
        
        total_event_profit <- total_event_profit + profit
        
        # Update bet record with outcome
        event_bets[j, "won"] <- won_bet
        event_bets[j, "profit"] <- profit
      }
      
      # Update bankroll after all bets in event are resolved
      bankroll <- bankroll + total_event_profit
      
      # Add to overall bet history
      bet_history <- rbind(bet_history, event_bets)
      
      # Record event summary
      event_summary <- data.frame(
        event_number = event_idx,
        EventID = current_event,
        bankroll_start = event_start_bankroll,
        bankroll_end = bankroll,
        event_profit = total_event_profit,
        cumulative_profit = bankroll - initial_bankroll,
        bets_in_event = bet_counter_for_event,
        total_bet_amount = event_total_bet,
        stringsAsFactors = FALSE
      )
      
    } else {
      # No bets placed in this event
      event_summary <- data.frame(
        event_number = event_idx,
        EventID = current_event,
        bankroll_start = event_start_bankroll,
        bankroll_end = bankroll,
        event_profit = 0,
        cumulative_profit = bankroll - initial_bankroll,
        bets_in_event = 0,
        total_bet_amount = 0,
        stringsAsFactors = FALSE
      )
    }
    
    bankroll_history <- rbind(bankroll_history, event_summary)
  }
  
  # Calculate summary statistics
  if(nrow(bet_history) > 0) {
    win_rate <- mean(bet_history$won, na.rm = TRUE)
    avg_bet_size <- mean(bet_history$bet_amount, na.rm = TRUE)
    total_wagered <- sum(bet_history$bet_amount, na.rm = TRUE)
    avg_odds <- mean(bet_history$odds, na.rm = TRUE)
    avg_prob_edge <- mean(bet_history$prob_edge, na.rm = TRUE)
  } else {
    win_rate <- avg_bet_size <- total_wagered <- avg_odds <- avg_prob_edge <- 0
  }
  
  return(list(
    bet_history = bet_history,
    bankroll_history = bankroll_history,
    final_bankroll = bankroll,
    total_profit = bankroll - initial_bankroll,
    roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100,
    total_bets = total_bet_counter,
    win_rate = win_rate,
    avg_bet_size = avg_bet_size,
    total_wagered = total_wagered,
    avg_odds = avg_odds,
    avg_prob_edge = avg_prob_edge,
    parameters = list(
      initial_bankroll = initial_bankroll,
      kelly_multiplier = kelly_multiplier,
      max_bet_amount = max_bet_amount,
      max_bet_pct = max_bet_pct,
      max_event_pct = max_event_pct,
      min_kelly_threshold = min_kelly_threshold,
      min_expected_value = min_expected_value,
      min_prob_edge = min_prob_edge,
      max_bets_per_event = max_bets_per_event,
      use_dynamic_bankroll = use_dynamic_bankroll
    )
  ))
}

# Function to print simulation results
print_simulation_results <- function(sim_result) {
  params <- sim_result$parameters
  
  cat("Kelly Betting Simulation Results\n")
  cat("================================\n")
  cat("Parameters:\n")
  cat("  Kelly Multiplier:", params$kelly_multiplier, 
      paste0("(", ifelse(params$kelly_multiplier == 1, "Full", 
                         ifelse(params$kelly_multiplier == 0.5, "Half", 
                                paste0(params$kelly_multiplier * 100, "%"))), " Kelly)\n"))
  cat("  Initial Bankroll: £", params$initial_bankroll, "\n")
  cat("  Max Bet Amount: £", params$max_bet_amount, "\n")
  cat("  Max Bet %:", params$max_bet_pct * 100, "%\n")
  cat("  Max Event %:", params$max_event_pct * 100, "%\n")
  cat("  Min Kelly Threshold:", params$min_kelly_threshold * 100, "%\n")
  cat("  Min Expected Value:", params$min_expected_value * 100, "%\n")
  cat("  Min Probability Edge:", params$min_prob_edge * 100, "%\n")
  cat("  Max Bets Per Event:", params$max_bets_per_event, "\n")
  cat("  Dynamic Bankroll:", params$use_dynamic_bankroll, "\n\n")
  
  cat("Results:\n")
  cat("  Final Bankroll: £", round(sim_result$final_bankroll, 2), "\n")
  cat("  Total Profit: £", round(sim_result$total_profit, 2), "\n")
  cat("  ROI:", round(sim_result$roi, 2), "%\n")
  cat("  Total Bets:", sim_result$total_bets, "\n")
  
  if(sim_result$total_bets > 0) {
    cat("  Win Rate:", round(sim_result$win_rate * 100, 2), "%\n")
    cat("  Average Bet Size: £", round(sim_result$avg_bet_size, 2), "\n")
    cat("  Total Wagered: £", round(sim_result$total_wagered, 2), "\n")
    cat("  Average Odds:", round(sim_result$avg_odds, 2), "\n")
    cat("  Average Probability Edge:", round(sim_result$avg_prob_edge * 100, 2), "%\n")
    
    # Betting constraint analysis
    if(nrow(sim_result$bet_history) > 0) {
      capped_amount <- sum(sim_result$bet_history$bet_capped_amount, na.rm = TRUE)
      capped_pct <- sum(sim_result$bet_history$bet_capped_pct, na.rm = TRUE)
      event_scaled <- sum(sim_result$bet_history$event_scaled, na.rm = TRUE)
      
      cat("  Bets capped by amount:", capped_amount, "\n")
      cat("  Bets capped by percentage:", capped_pct, "\n")
      cat("  Bets scaled by event budget:", event_scaled, "\n")
    }
  }
}

# Function to plot bankroll sims with parameters
plot_with_param_table <- function(sim_results_list, sim_names = NULL) {
  
  library(ggplot2)
  library(dplyr)
  library(gridExtra)
  library(grid)
  
  # If no names provided, create default names
  if(is.null(sim_names)) {
    sim_names <- paste("Simulation", 1:length(sim_results_list))
  }
  
  # Create the main plot (same as before)
  combined_data <- data.frame()
  
  for(i in 1:length(sim_results_list)) {
    sim_result <- sim_results_list[[i]]
    bankroll_history <- sim_result$bankroll_history
    plot_data <- bankroll_history[bankroll_history$event_number > 0, ]
    plot_data$simulation <- sim_names[i]
    plot_data$initial_bankroll <- sim_result$parameters$initial_bankroll
    combined_data <- rbind(combined_data, plot_data)
  }
  
  main_plot <- ggplot(combined_data, aes(x = event_number, y = bankroll_end, color = simulation)) +
    geom_line(size = 0.8) +
    geom_point(size = 1.5, alpha = 0.7) +
    geom_hline(data = combined_data %>% 
                 group_by(simulation) %>% 
                 summarise(initial_bankroll = first(initial_bankroll), .groups = 'drop'),
               aes(yintercept = initial_bankroll, color = simulation),
               linetype = "dashed", alpha = 0.7) +
    labs(x = "Event Number", 
         y = "Bankroll (£)",
         color = "Simulation") +
    theme_minimal() +
    theme(legend.position = "none")  # Remove legend as info will be in table
  
  # Create comprehensive parameter comparison table
  param_df <- data.frame()
  for(i in 1:length(sim_results_list)) {
    sim_result <- sim_results_list[[i]]
    params <- sim_result$parameters
    
    param_row <- data.frame(
      Simulation = sim_names[i],
      `Initial £` = params$initial_bankroll,
      `Kelly %` = paste0(params$kelly_multiplier * 100, "%"),
      `Max Bet £` = params$max_bet_amount,
      `Max Bet %` = paste0(params$max_bet_pct * 100, "%"),
      `Max Event %` = paste0(params$max_event_pct * 100, "%"),
      `Min Kelly %` = paste0(params$min_kelly_threshold * 100, "%"),
      `Min EV %` = paste0(params$min_expected_value * 100, "%"),
      `Min Edge %` = paste0(params$min_prob_edge * 100, "%"),
      `Max Bets/Event` = params$max_bets_per_event,
      `Dynamic BR` = ifelse(params$use_dynamic_bankroll, "Yes", "No"),
      `Final £` = round(sim_result$final_bankroll, 0),
      `ROI %` = paste0(round(sim_result$roi, 1), "%"),
      `Total Bets` = sim_result$total_bets,
      `Win Rate %` = paste0(round(sim_result$win_rate * 100, 1), "%"),
      check.names = FALSE
    )
    param_df <- rbind(param_df, param_row)
  }
  
  # Create table grob
  table_grob <- tableGrob(param_df, rows = NULL, 
                          theme = ttheme_minimal(base_size = 9))
  
  # Combine plot and table
  combined_plot <- grid.arrange(main_plot, table_grob, 
                                heights = c(3, 1), 
                                ncol = 1)
  
  return(combined_plot)
}

# Customised Kelly Sim
custom_kelly_sim_2023 <- simulate_kelly_betting_custom(
  new_preds_kelly_2023,
  initial_bankroll = 1000,
  kelly_multiplier = 0.5,       # Fraction of Kelly to bet (e.g. 0.5 = half Kelly)
  max_bet_amount = 500,           # Maximum bet in currency units
  max_bet_pct = 1,            # Maximum bet as % of bankroll
  max_event_pct = 1,          # Maximum total bets per event as % of bankroll
  min_kelly_threshold = 0.0,    # Minimum Kelly fraction to consider betting
  min_expected_value = 0,     # Minimum expected value to bet
  min_prob_edge = 0.06,          # Minimum probability edge over market
  max_bets_per_event = 10,        # Maximum number of bets per event
  use_dynamic_bankroll = F,
  min_odds = 1,
  max_odds = 999
)

custom_kelly_sim_2024 <- simulate_kelly_betting_custom(
  new_preds_kelly_99,
  initial_bankroll = 1000,
  kelly_multiplier = 0.5,       # Fraction of Kelly to bet (e.g. 0.5 = half Kelly)
  max_bet_amount = 500,           # Maximum bet in currency units
  max_bet_pct = 1,            # Maximum bet as % of bankroll
  max_event_pct = 1,          # Maximum total bets per event as % of bankroll
  min_kelly_threshold = 0.0,    # Minimum Kelly fraction to consider betting
  min_expected_value = 0,     # Minimum expected value to bet
  min_prob_edge = 0.06,          # Minimum probability edge over market
  max_bets_per_event = 10,        # Maximum number of bets per event
  use_dynamic_bankroll = F,
  min_odds = 1,
  max_odds = 999
)

# Print simulation results
print_simulation_results(custom_kelly_sim_2023)
print_simulation_results(custom_kelly_sim_2024)

# Plot simulation bankroll
combined_plot <- plot_with_param_table(
  list(custom_kelly_sim_2023, custom_kelly_sim_2024),
  c("2023", "2024")
)

# Save simulation results
write.csv(custom_kelly_sim_2023[["bet_history"]], "./Betting Simulations/Bet_History_23.csv", row.names=F)
write.csv(custom_kelly_sim_2023[["bankroll_history"]], "./Betting Simulations/Bankroll_History_23.csv", row.names=F)

write.csv(custom_kelly_sim_2024[["bet_history"]], "./Betting Simulations/Bet_History_24.csv", row.names=F)
write.csv(custom_kelly_sim_2024[["bankroll_history"]], "./Betting Simulations/Bankroll_History_24.csv", row.names=F)

write.csv(custom_kelly_sim_2023[["parameters"]], "./Betting Simulations/Simulation_Parameters.csv", row.names=F)
