library(dplyr)
library(ggplot2)
library(gridExtra)

# Function to run multiple Kelly configurations and compare results
compare_kelly_configurations <- function(df, configurations) {
  
  results_list <- list()
  results_summary <- data.frame()
  
  # Run simulation for each configuration
  for(i in 1:length(configurations)) {
    config_name <- names(configurations)[i]
    config <- configurations[[i]]
    
    cat("Running simulation:", config_name, "\n")
    
    # Run the simulation with current configuration
    sim_result <- do.call(simulate_kelly_betting_custom, c(list(df = df), config))
    
    # Store full results
    results_list[[config_name]] <- sim_result
    
    # Create summary row
    summary_row <- data.frame(
      Configuration = config_name,
      Kelly_Multiplier = config$kelly_multiplier,
      Min_Kelly_Threshold = config$min_kelly_threshold,
      Min_Expected_Value = config$min_expected_value,
      Min_Prob_Edge = config$min_prob_edge,
      Max_Bets_Per_Event = config$max_bets_per_event,
      Max_Event_Pct = config$max_event_pct,
      Initial_Bankroll = config$initial_bankroll,
      Final_Bankroll = sim_result$final_bankroll,
      Total_Profit = sim_result$total_profit,
      ROI_Percent = sim_result$roi,
      Total_Bets = sim_result$total_bets,
      Win_Rate_Percent = sim_result$win_rate * 100,
      Avg_Bet_Size = sim_result$avg_bet_size,
      Total_Wagered = sim_result$total_wagered,
      Avg_Odds = sim_result$avg_odds,
      Avg_Prob_Edge_Percent = sim_result$avg_prob_edge * 100,
      Risk_Reward_Ratio = ifelse(sim_result$total_wagered > 0, 
                                 sim_result$total_profit / sim_result$total_wagered, 0),
      Profit_Per_Bet = ifelse(sim_result$total_bets > 0, 
                              sim_result$total_profit / sim_result$total_bets, 0),
      stringsAsFactors = FALSE
    )
    
    results_summary <- rbind(results_summary, summary_row)
  }
  
  return(list(
    detailed_results = results_list,
    summary = results_summary
  ))
}

# Function to create comparison visualizations
create_comparison_plots <- function(comparison_results) {
  
  summary_df <- comparison_results$summary
  detailed_results <- comparison_results$detailed_results
  
  plots_list <- list()
  
  # 1. ROI Comparison
  p1 <- ggplot(summary_df, aes(x = reorder(Configuration, ROI_Percent), y = ROI_Percent, fill = Configuration)) +
    geom_col() +
    geom_text(aes(label = paste0(round(ROI_Percent, 1), "%")), 
              hjust = ifelse(summary_df$ROI_Percent >= 0, -0.1, 1.1)) +
    coord_flip() +
    labs(title = "ROI Comparison Across Configurations",
         x = "Configuration", y = "ROI (%)") +
    theme_minimal() +
    theme(legend.position = "none")
  
  plots_list[["roi_comparison"]] <- p1
  
  # 2. Risk vs Return Scatter
  p2 <- ggplot(summary_df, aes(x = Total_Wagered, y = Total_Profit, color = Configuration, size = Total_Bets)) +
    geom_point(alpha = 0.7) +
    geom_text(aes(label = Configuration), vjust = -1, size = 3) +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    labs(title = "Risk vs Return Analysis",
         x = "Total Amount Wagered (£)", y = "Total Profit (£)",
         size = "Number of Bets") +
    theme_minimal()
  
  plots_list[["risk_return"]] <- p2
  
  # 3. Total Bets vs ROI
  p3 <- ggplot(summary_df, aes(x = Total_Bets, y = ROI_Percent, color = Configuration)) +
    geom_point(size = 3) +
    geom_text(aes(label = Configuration), vjust = -1, size = 3) +
    labs(title = "Number of Bets vs ROI",
         x = "Total Number of Bets", y = "ROI (%)") +
    theme_minimal()
  
  plots_list[["bets_vs_roi"]] <- p3
  
  # 4. Bankroll Evolution Over Time (if multiple configs)
  if(length(detailed_results) > 1) {
    # Combine bankroll histories
    combined_bankroll <- data.frame()
    
    for(config_name in names(detailed_results)) {
      bankroll_hist <- detailed_results[[config_name]]$bankroll_history
      if(nrow(bankroll_hist) > 0) {
        bankroll_hist$Configuration <- config_name
        combined_bankroll <- rbind(combined_bankroll, bankroll_hist)
      }
    }
    
    if(nrow(combined_bankroll) > 0) {
      p4 <- ggplot(combined_bankroll, aes(x = event_number, y = bankroll_end, color = Configuration)) +
        geom_line(size = 1) +
        geom_hline(yintercept = unique(combined_bankroll$bankroll_start)[1], 
                   linetype = "dashed", alpha = 0.5) +
        labs(title = "Bankroll Evolution Over Time",
             x = "Event Number", y = "Bankroll (£)") +
        theme_minimal() +
        theme(legend.position = "bottom")
      
      plots_list[["bankroll_evolution"]] <- p4
    }
  }
  
  # 5. Win Rate vs Average Bet Size
  p5 <- ggplot(summary_df, aes(x = Win_Rate_Percent, y = Avg_Bet_Size, color = Configuration, size = Total_Bets)) +
    geom_point(alpha = 0.7) +
    geom_text(aes(label = Configuration), vjust = -1, size = 3) +
    labs(title = "Win Rate vs Average Bet Size",
         x = "Win Rate (%)", y = "Average Bet Size (£)",
         size = "Total Bets") +
    theme_minimal()
  
  plots_list[["winrate_vs_betsize"]] <- p5
  
  return(plots_list)
}

# Function to print detailed comparison
print_comparison_summary <- function(comparison_results) {
  
  summary_df <- comparison_results$summary
  
  cat("KELLY CONFIGURATION COMPARISON SUMMARY\n")
  cat("=====================================\n\n")
  
  # Rank by ROI
  summary_ranked <- summary_df %>% arrange(desc(ROI_Percent))
  
  cat("CONFIGURATIONS RANKED BY ROI:\n")
  cat("-----------------------------\n")
  for(i in 1:nrow(summary_ranked)) {
    row <- summary_ranked[i, ]
    cat(sprintf("%d. %s: %.1f%% ROI (£%.0f profit, %d bets, %.1f%% win rate)\n",
                i, row$Configuration, row$ROI_Percent, row$Total_Profit, 
                row$Total_Bets, row$Win_Rate_Percent))
  }
  
  cat("\nKEY METRICS COMPARISON:\n")
  cat("-----------------------\n")
  cat(sprintf("Best ROI: %s (%.1f%%)\n", 
              summary_ranked$Configuration[1], summary_ranked$ROI_Percent[1]))
  cat(sprintf("Highest Profit: %s (£%.0f)\n", 
              summary_df$Configuration[which.max(summary_df$Total_Profit)], 
              max(summary_df$Total_Profit)))
  cat(sprintf("Most Bets: %s (%d bets)\n", 
              summary_df$Configuration[which.max(summary_df$Total_Bets)], 
              max(summary_df$Total_Bets)))
  cat(sprintf("Highest Win Rate: %s (%.1f%%)\n", 
              summary_df$Configuration[which.max(summary_df$Win_Rate_Percent)], 
              max(summary_df$Win_Rate_Percent)))
  cat(sprintf("Lowest Risk: %s (£%.0f wagered)\n", 
              summary_df$Configuration[which.min(summary_df$Total_Wagered)], 
              min(summary_df$Total_Wagered)))
  
  cat("\nDETAILED RESULTS TABLE:\n")
  cat("-----------------------\n")
  print(summary_df %>% 
          select(Configuration, ROI_Percent, Total_Profit, Total_Bets, 
                 Win_Rate_Percent, Avg_Bet_Size, Total_Wagered) %>%
          arrange(desc(ROI_Percent)))
}

# Predefined configuration sets
get_predefined_configurations <- function() {
  
  configurations <- list(
    
    "Conservative" = list(
      initial_bankroll = 1000,
      kelly_multiplier = 0.25,
      max_bet_amount = 25,
      max_bet_pct = 0.025,
      max_event_pct = 0.15,
      min_kelly_threshold = 0.06,
      min_expected_value = 0.12,
      min_prob_edge = 0.08,
      max_bets_per_event = 2,
      use_dynamic_bankroll = TRUE
    ),
    
    "Moderate" = list(
      initial_bankroll = 1000,
      kelly_multiplier = 0.5,
      max_bet_amount = 50,
      max_bet_pct = 0.05,
      max_event_pct = 0.30,
      min_kelly_threshold = 0.04,
      min_expected_value = 0.08,
      min_prob_edge = 0.05,
      max_bets_per_event = 5,
      use_dynamic_bankroll = TRUE
    ),
    
    "Aggressive" = list(
      initial_bankroll = 1000,
      kelly_multiplier = 0.75,
      max_bet_amount = 75,
      max_bet_pct = 0.08,
      max_event_pct = 0.50,
      min_kelly_threshold = 0.02,
      min_expected_value = 0.05,
      min_prob_edge = 0.03,
      max_bets_per_event = 8,
      use_dynamic_bankroll = TRUE
    ),
    
    "High_Volume" = list(
      initial_bankroll = 1000,
      kelly_multiplier = 0.3,
      max_bet_amount = 30,
      max_bet_pct = 0.03,
      max_event_pct = 0.40,
      min_kelly_threshold = 0.015,
      min_expected_value = 0.04,
      min_prob_edge = 0.02,
      max_bets_per_event = 12,
      use_dynamic_bankroll = TRUE
    ),
    
    "Quality_Focus" = list(
      initial_bankroll = 1000,
      kelly_multiplier = 0.6,
      max_bet_amount = 100,
      max_bet_pct = 0.10,
      max_event_pct = 0.25,
      min_kelly_threshold = 0.08,
      min_expected_value = 0.15,
      min_prob_edge = 0.10,
      max_bets_per_event = 3,
      use_dynamic_bankroll = TRUE
    )
  )
  
  return(configurations)
}

# Run comparison with predefined configurations
predefined_configs <- get_predefined_configurations()
comparison_results <- compare_kelly_configurations(new_preds_kelly, predefined_configs)

# Print summary
print_comparison_summary(comparison_results)

# Create and display plots
plots <- create_comparison_plots(comparison_results)
grid.arrange(plots$roi_comparison, 
             plots$risk_return,
             plots$bankroll_evolution, 
             ncol = 2)

# Access detailed results for specific configuration
# conservative_detailed <- comparison_results$detailed_results$Conservative
# print_simulation_results(conservative_detailed)

# Create custom configurations for comparison
custom_configs <- list(
"Custom_1" = list(
  initial_bankroll = 1000,
  kelly_multiplier = 0.5,
  max_bet_amount = 100,
  max_bet_pct = 0.10,
  max_event_pct = 0.25,
  min_kelly_threshold = 0.08,
  min_expected_value = 0.15,
  min_prob_edge = 0.10,
  max_bets_per_event = 3,
  use_dynamic_bankroll = FALSE
),

"Custom_2" = list(
  initial_bankroll = 1000,
  kelly_multiplier = 0.5,
  max_bet_amount = 100,
  max_bet_pct = 0.05,
  max_event_pct = 0.4,
  min_kelly_threshold = 0.04,
  min_expected_value = 0.10,
  min_prob_edge = 0.10,
  max_bets_per_event = 10,
  use_dynamic_bankroll = FALSE
)
)

custom_comparison <- compare_kelly_configurations(new_preds_kelly, custom_configs)
plots <- create_comparison_plots(custom_comparison)


print_comparison_summary(custom_comparison)
grid.arrange(plots$roi_comparison, 
             plots$risk_return,
             plots$bankroll_evolution, 
             ncol = 2)
