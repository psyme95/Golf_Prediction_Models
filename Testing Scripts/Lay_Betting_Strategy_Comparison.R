#### Packages ####
library(dplyr)
library(ggplot2)
library(gridExtra)  # Added for combining plots
library(tidyr)      # Added for pivot_longer

#### Load Model Predictions ####
all_predictions <- read.csv("C:/Projects/Golf/Results/Top20_RW_1607_0117/Model_Predictions.csv") # Needs updating, run "apply_different_calibration_techniques" script for platt scaling

#### Fixed Parameters ####
Starting_Bankroll <- 1000
Fixed_Stake <- 5

# Define the different filter types and thresholds to test
filter_configs <- list(
  model_score = list(
    thresholds = seq(50, 1000, by = 50),
    filter_column = "Model_Score",
    filter_operator = "<",
    filter_name = "Model Score"
  ),
  model_rank = list(
    thresholds = seq(5, 160, by = 5),
    filter_column = "Model_Rank",
    filter_operator = "<=",
    filter_name = "Model Rank"
  ),
  probability = list(
    thresholds = seq(0.05, 0.5, by = 0.01),
    filter_column = "Calibrated_Probability",
    filter_operator = "<=",
    filter_name = "Calibrated Probability"
  )
)

# Initialize list to store all plots and Kelly results
all_plots <- list()
all_kelly_results <- list()

#### Simulation Functions ####
# Dynamic Kelly Criterion simulation function
calculate_kelly_lay_dynamic_bankroll <- function(df, kelly_modifier = 0.1, initial_bankroll = 1000, max_event_exposure = 1) {
  
  # Sort by EventID to process chronologically
  df <- df[order(df$EventID), ]
  
  # Get unique events in order
  unique_events <- unique(df$EventID)
  
  # Initialize results dataframe
  results_list <- list()
  
  # Track bankroll progression
  bankroll_history <- data.frame(
    EventID = numeric(),
    Starting_Bankroll = numeric(),
    Event_PL = numeric(),
    Ending_Bankroll = numeric()
  )
  
  current_bankroll <- initial_bankroll
  
  # Process each event
  for (i in seq_along(unique_events)) {
    event_id <- unique_events[i]
    event_data <- df[df$EventID == event_id, ]
    
    # Calculate Kelly fractions for this event
    event_data$Kelly_Fraction <- with(event_data, {
      win_prob <- 1 - Calibrated_Probability
      lose_prob <- Calibrated_Probability
      net_odds <- Lay_Odds - 1
      kelly_fraction <- (net_odds * win_prob - lose_prob) / net_odds
      return(kelly_fraction)
    })
    
    # Apply modifier and set negative Kelly to 0
    event_data$Modified_Kelly <- pmax(event_data$Kelly_Fraction * kelly_modifier, 0)
    
    # Calculate unconstrained stakes and liability based on current bankroll
    event_data$Current_Bankroll <- current_bankroll
    event_data$Unconstrained_Stake <- event_data$Modified_Kelly * current_bankroll
    event_data$Unconstrained_Liability <- event_data$Unconstrained_Stake * (event_data$Lay_Odds - 1)
    
    # Apply event-level liability constraint
    event_total_unconstrained_liability <- sum(event_data$Unconstrained_Liability, na.rm = TRUE)
    event_max_liability <- max_event_exposure * current_bankroll
    
    scaling_factor <- ifelse(event_total_unconstrained_liability > event_max_liability,
                             event_max_liability / event_total_unconstrained_liability,
                             1)
    
    # Apply scaling
    event_data$Scaling_Factor <- scaling_factor
    event_data$Stake <- event_data$Unconstrained_Stake * scaling_factor
    event_data$Liability <- event_data$Unconstrained_Liability * scaling_factor
    
    # Calculate profit/loss for each bet
    event_data$Profit_Loss <- with(event_data, {
      ifelse(Actual_Top20 == 0, Stake, -Liability)
    })
    
    # Calculate event totals
    event_pl <- sum(event_data$Profit_Loss, na.rm = TRUE)
    
    # Update bankroll
    new_bankroll <- current_bankroll + event_pl
    
    # Store bankroll history
    bankroll_history <- rbind(bankroll_history, data.frame(
      EventID = event_id,
      Starting_Bankroll = current_bankroll,
      Event_PL = event_pl,
      Ending_Bankroll = new_bankroll
    ))
    
    # Update current bankroll for next event
    current_bankroll <- new_bankroll
    
    # Store results
    results_list[[i]] <- event_data
  }
  
  # Combine all results
  final_results <- do.call(rbind, results_list)
  
  # Add bankroll history to the package
  return(list(
    results = final_results,
    bankroll_history = bankroll_history
  ))
}

# Function to run dynamic bankroll analysis for a given threshold and filter type
run_dynamic_bankroll <- function(threshold, data, filter_column, filter_operator) {
  
  # Filter data by the specified threshold and column
  if(filter_operator == "<") {
    filtered_data <- data %>%
      select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
      filter(!!sym(filter_column) < threshold) %>%
      mutate(Lay_Odds = Top20_odds * 1.1) %>%
      arrange(EventNumber)
  } else if(filter_operator == "<=") {
    filtered_data <- data %>%
      select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
      filter(!!sym(filter_column) <= threshold) %>%
      mutate(Lay_Odds = Top20_odds * 1.1) %>%
      arrange(EventNumber)
  } else if(filter_operator == ">") {
    filtered_data <- data %>%
      select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
      filter(!!sym(filter_column) > threshold) %>%
      mutate(Lay_Odds = Top20_odds * 1.1) %>%
      arrange(EventNumber)
  } else if(filter_operator == ">=") {
    filtered_data <- data %>%
      select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
      filter(!!sym(filter_column) >= threshold) %>%
      mutate(Lay_Odds = Top20_odds * 1.1) %>%
      arrange(EventNumber)
  }
  
  # Check if we have any data after filtering
  if(nrow(filtered_data) == 0) {
    return(data.frame(
      Threshold = threshold,
      Start_Bankroll = Starting_Bankroll,
      Final_Bankroll_FS = Starting_Bankroll,
      Final_Bankroll_FL = Starting_Bankroll,
      Fixed_Stake_Profit = 0,
      Fixed_Liability_Profit = 0,
      Total_Events = 0,
      Total_Bets = 0,
      Avg_Bets_Per_Event = 0,
      ROI_FS = 0,
      ROI_FL = 0
    ))
  }
  
  # Get unique events in order
  events <- unique(filtered_data$EventNumber)
  
  # Initialize results list
  results_list <- list()
  current_bankroll_fs <- Starting_Bankroll
  current_bankroll_fl <- Starting_Bankroll
  
  # Process each event sequentially
  for(event in events) {
    
    # Filter data for current event
    event_data <- filtered_data %>%
      filter(EventNumber == event)
    
    # Skip if no bets for this event
    if(nrow(event_data) == 0) next
    
    # Calculate stakes and outcomes based on current bankroll
    event_results <- event_data %>%
      mutate(
        Current_Bankroll_FS = current_bankroll_fs,
        Current_Bankroll_FL = current_bankroll_fl,
        Field = n(),
        
        # Fixed Stake approach (using current bankroll)
        Fixed_Stake = Fixed_Stake,
        FS_Liability = Fixed_Stake * (Lay_Odds - 1),
        Total_FS_Liability = sum(FS_Liability),
        FS_Multiplier = current_bankroll_fs / Total_FS_Liability,
        FS_Adjusted = Fixed_Stake * FS_Multiplier,
        FS_Liability_Adjusted = FS_Adjusted * (Lay_Odds - 1),
        FS_Outcome = ifelse(Actual_Top20 == 0, FS_Adjusted, -FS_Liability_Adjusted),
        
        # Fixed Liability approach (using current bankroll)
        Fixed_Liability = round(current_bankroll_fl / Field, 2),
        FL_Stake = Fixed_Liability / (Lay_Odds - 1),
        FL_Outcome = ifelse(Actual_Top20 == 0, FL_Stake, -Fixed_Liability)
      )
    
    # Store results
    results_list[[length(results_list) + 1]] <- event_results
    
    # Update bankroll for next event (separate for each strategy)
    current_bankroll_fs <- current_bankroll_fs + sum(event_results$FS_Outcome)
    current_bankroll_fl <- current_bankroll_fl + sum(event_results$FL_Outcome)
    
    # Optional: Add bankroll protection
    if(current_bankroll_fs <= 0 || current_bankroll_fl <= 0) {
      warning(paste("Bankroll depleted at event", event, "for threshold", threshold))
      break
    }
  }
  
  # Combine all results
  if(length(results_list) == 0) {
    return(data.frame(
      Threshold = threshold,
      Start_Bankroll = Starting_Bankroll,
      Final_Bankroll_FS = Starting_Bankroll,
      Final_Bankroll_FL = Starting_Bankroll,
      Fixed_Stake_Profit = 0,
      Fixed_Liability_Profit = 0,
      Total_Events = 0,
      Total_Bets = 0,
      Avg_Bets_Per_Event = 0,
      ROI_FS = 0,
      ROI_FL = 0
    ))
  }
  
  all_results <- bind_rows(results_list)
  
  # Calculate event-level summaries
  event_summaries <- all_results %>%
    group_by(EventNumber) %>%
    summarise(
      Event_FS_Profit = sum(FS_Outcome),
      Event_FL_Profit = sum(FL_Outcome),
      Bets_Count = n(),
      .groups = 'drop'
    ) %>%
    arrange(EventNumber) %>%
    mutate(
      Cumulative_FS_Profit = cumsum(Event_FS_Profit),
      Cumulative_FL_Profit = cumsum(Event_FL_Profit),
      Running_Bankroll_FS = Starting_Bankroll + Cumulative_FS_Profit,
      Running_Bankroll_FL = Starting_Bankroll + Cumulative_FL_Profit
    )
  
  # Summary for this threshold
  summary_stats <- data.frame(
    Threshold = threshold,
    Start_Bankroll = Starting_Bankroll,
    Final_Bankroll_FS = last(event_summaries$Running_Bankroll_FS),
    Final_Bankroll_FL = last(event_summaries$Running_Bankroll_FL),
    Fixed_Stake_Profit = last(event_summaries$Cumulative_FS_Profit),
    Fixed_Liability_Profit = last(event_summaries$Cumulative_FL_Profit),
    Total_Events = nrow(event_summaries),
    Total_Bets = nrow(all_results),
    Avg_Bets_Per_Event = round(nrow(all_results) / nrow(event_summaries), 2),
    ROI_FS = round((last(event_summaries$Cumulative_FS_Profit) / Starting_Bankroll) * 100, 2),
    ROI_FL = round((last(event_summaries$Cumulative_FL_Profit) / Starting_Bankroll) * 100, 2)
  )
  
  return(summary_stats)
}

# Modified function to create plots without displaying them immediately
create_filter_plot <- function(filter_type, filtering_comparison) {
  
  config <- filter_configs[[filter_type]]
  filter_name <- config$filter_name
  
  # Create the plot
  plot_title <- paste("Final Bankroll vs", filter_name, "Threshold")
  x_label <- paste(filter_name, "Threshold")
  
  p <- filtering_comparison %>%
    select(Threshold, Final_Bankroll_FS, Final_Bankroll_FL) %>%
    pivot_longer(cols = c(Final_Bankroll_FS, Final_Bankroll_FL), names_to = "Strategy", values_to = "Bankroll") %>%
    ggplot(aes(x = Threshold, y = Bankroll, color = Strategy)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    geom_hline(yintercept = Starting_Bankroll, linetype = "dashed", alpha = 0.5) +
    labs(title = plot_title,
         subtitle = "Fixed Stake/Liability",
         x = x_label,
         y = "Final Bankroll (£)",
         color = "Strategy") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10),
      legend.position = "bottom"
    ) +
    scale_color_manual(values = c("Final_Bankroll_FS" = "blue", "Final_Bankroll_FL" = "red"),
                       labels = c("Fixed Stake", "Fixed Liability")) +
    ylim(0, 5000)
  
  # Adjust x-axis breaks based on filter type
  if(filter_type == "model_score") {
    p <- p + scale_x_continuous(breaks = seq(50, 900, by = 200))
  } else if(filter_type == "model_rank") {
    p <- p + scale_x_continuous(breaks = seq(5, 160, by = 40))
  } else if(filter_type == "probability") {
    p <- p + scale_x_continuous(breaks = seq(0, 0.5, by = 0.1))
  }
  
  return(p)
}

# Function to run analysis for a specific filter type (modified to not display plots)
run_filter_analysis <- function(filter_type, data) {
  
  config <- filter_configs[[filter_type]]
  thresholds <- config$thresholds
  filter_column <- config$filter_column
  filter_operator <- config$filter_operator
  filter_name <- config$filter_name
  
  cat("\n=== Running analysis for", filter_name, "===\n")
  
  # Initialize results storage
  filtering_results <- list()
  
  # Run the analysis
  for(i in seq_along(thresholds)) {
    threshold <- thresholds[i]
    cat("Processing", filter_name, "threshold:", threshold, "(", i, "of", length(thresholds), ")\n")
    
    result <- run_dynamic_bankroll(threshold, data, filter_column, filter_operator)
    result$Filter_Type <- filter_type
    result$Filter_Name <- filter_name
    filtering_results[[i]] <- result
  }
  
  # Combine all results
  filtering_comparison <- bind_rows(filtering_results)
  
  # Create plot but don't display it
  plot_obj <- create_filter_plot(filter_type, filtering_comparison)
  
  # Find and print optimal thresholds
  best_fs <- filtering_comparison[which.max(filtering_comparison$Final_Bankroll_FS), ]
  best_fl <- filtering_comparison[which.max(filtering_comparison$Final_Bankroll_FL), ]
  
  cat("\nOptimal Results for", filter_name, ":\n")
  cat("Fixed Stake - Threshold:", best_fs$Threshold, "Final Bankroll: £", best_fs$Final_Bankroll_FS, "\n")
  cat("Fixed Liability - Threshold:", best_fl$Threshold, "Final Bankroll: £", best_fl$Final_Bankroll_FL, "\n")
  
  return(list(
    data = filtering_comparison,
    plot = plot_obj
  ))
}

# Modified Kelly plotting function
create_kelly_plot <- function(results_df, filter_name, x_var) {
  p <- ggplot(results_df, aes(x = !!sym(x_var), y = Final_Bankroll)) +
    geom_line(color = "blue", size = 1) +
    geom_point(color = "red", size = 2) +
    geom_hline(yintercept = 1000, linetype = "dashed", color = "black", alpha = 0.7) +
    labs(
      title = paste("Final Bankroll vs", filter_name, "Filter"),
      subtitle = "Kelly Criterion",
      x = paste(filter_name, "Filter"),
      y = "Final Bankroll (£)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 10)
    ) +
    ylim(0, 5000)
  
  # Adjust x-axis breaks based on filter type
  if(filter_name == "Model Score") {
    p <- p + scale_x_continuous(breaks = seq(50, 900, by = 200))
  } else if(filter_name == "Model Rank") {
    p <- p + scale_x_continuous(breaks = seq(5, 160, by = 40))
  } else if(filter_name == "Probability") {
    p <- p + scale_x_continuous(breaks = seq(0, 0.5, by = 0.1))
  }
  
  return(p)
}

#### Fixed bankroll - Fixed Stake & liability outcomes - No filtering ####
all_predictions_lay <- all_predictions_platt %>%
  dplyr::select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
  mutate(Lay_Odds = Top20_odds * 1.1,
         Fixed_Stake = Fixed_Stake,
         FS_Liability = Fixed_Stake * (Lay_Odds - 1)) %>%
  group_by(EventNumber) %>%
  mutate(Field = n(),
         Fixed_Liability = round(Starting_Bankroll / Field, 2),
         FL_Stake = Fixed_Liability / (Lay_Odds - 1),
         FL_Outcome = ifelse(Actual_Top20 == 0,
                             FL_Stake,
                             -Fixed_Liability),
         Total_FS_Liability = sum(FS_Liability),
         FS_Multiplier = Starting_Bankroll / Total_FS_Liability,
         FS_Adjusted = Fixed_Stake * FS_Multiplier,
         FS_Liability_Adjusted = FS_Adjusted * (Lay_Odds - 1),
         FS_Outcome = ifelse(Actual_Top20 == 0,
                             FS_Adjusted,
                             -FS_Liability_Adjusted))

Outcome_Summary <- all_predictions_lay %>%
  ungroup() %>%
  summarise(Start_Bankroll = Starting_Bankroll,
            Fixed_Stake_Profit = sum(FS_Outcome),
            Fixed_Liability_Profit = sum(FL_Outcome))
print(Outcome_Summary)

#### Fixed bankroll - Fixed Stake & liability outcomes - Filtering ####
all_predictions_lay <- all_predictions_platt %>%
  dplyr::select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
  filter(Model_Score < 100) %>%
  mutate(Lay_Odds = Top20_odds * 1.1,
         Fixed_Stake = Fixed_Stake,
         FS_Liability = Fixed_Stake * (Lay_Odds - 1)) %>%
  group_by(EventNumber) %>%
  mutate(Field = n(),
         Fixed_Liability = round(Starting_Bankroll / Field, 2),
         FL_Stake = Fixed_Liability / (Lay_Odds - 1),
         FL_Outcome = ifelse(Actual_Top20 == 0,
                             FL_Stake,
                             -Fixed_Liability),
         Total_FS_Liability = sum(FS_Liability),
         FS_Multiplier = Starting_Bankroll / Total_FS_Liability,
         FS_Adjusted = Fixed_Stake * FS_Multiplier,
         FS_Liability_Adjusted = FS_Adjusted * (Lay_Odds - 1),
         FS_Outcome = ifelse(Actual_Top20 == 0,
                             FS_Adjusted,
                             -FS_Liability_Adjusted))

Outcome_Summary <- all_predictions_lay %>%
  ungroup() %>%
  summarise(Start_Bankroll = Starting_Bankroll,
            Fixed_Stake_Profit = sum(FS_Outcome),
            Fixed_Liability_Profit = sum(FL_Outcome))
print(Outcome_Summary)

#### Dynamic bankroll - Fixed Stake & liability - Single filter or no filter ####
all_predictions_lay <- all_predictions_platt %>%
  dplyr::select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
  mutate(Lay_Odds = Top20_odds * 1.1) %>%
  filter(Model_Score < 999) %>% # Filter to test/remove
  arrange(EventNumber)

# Get unique events in order
events <- unique(all_predictions_lay$EventNumber)

# Initialize results list
results_list <- list()
current_bankroll <- Starting_Bankroll

# Process each event sequentially
for(event in events) {
  
  # Filter data for current event
  event_data <- all_predictions_lay %>%
    filter(EventNumber == event)
  
  # Calculate stakes and outcomes based on current bankroll
  event_results <- event_data %>%
    mutate(
      Current_Bankroll = current_bankroll,
      Field = n(),
      
      # Fixed Stake approach (using current bankroll)
      Fixed_Stake = Fixed_Stake,
      FS_Liability = Fixed_Stake * (Lay_Odds - 1),
      Total_FS_Liability = sum(FS_Liability),
      FS_Multiplier = current_bankroll / Total_FS_Liability,
      FS_Adjusted = Fixed_Stake * FS_Multiplier,
      FS_Liability_Adjusted = FS_Adjusted * (Lay_Odds - 1),
      FS_Outcome = ifelse(Actual_Top20 == 0, FS_Adjusted, -FS_Liability_Adjusted),
      
      # Fixed Liability approach (using current bankroll)
      Fixed_Liability = round(current_bankroll / Field, 2),
      FL_Stake = Fixed_Liability / (Lay_Odds - 1),
      FL_Outcome = ifelse(Actual_Top20 == 0, FL_Stake, -Fixed_Liability)
    )
  
  # Store results
  results_list[[length(results_list) + 1]] <- event_results
  
  # Update bankroll for next event (sum once per event)
  current_bankroll <- current_bankroll + sum(event_results$FS_Outcome)  # or FL_Outcome
  
  # Optional: Add bankroll protection
  if(current_bankroll <= 0) {
    warning(paste("Bankroll depleted at event", event))
    break
  }
}

# Combine all results
all_predictions_lay_dynamic <- bind_rows(results_list)

# Calculate event-level summaries first
event_summaries <- all_predictions_lay_dynamic %>%
  group_by(EventNumber) %>%
  summarise(
    Event_FS_Profit = sum(FS_Outcome),
    Event_FL_Profit = sum(FL_Outcome),
    .groups = 'drop'
  ) %>%
  arrange(EventNumber) %>%
  mutate(
    Cumulative_FS_Profit = cumsum(Event_FS_Profit),
    Cumulative_FL_Profit = cumsum(Event_FL_Profit),
    Running_Bankroll_FS = Starting_Bankroll + Cumulative_FS_Profit,
    Running_Bankroll_FL = Starting_Bankroll + Cumulative_FL_Profit
  )

# Join back to main data
all_predictions_lay_dynamic <- all_predictions_lay_dynamic %>%
  left_join(event_summaries, by = "EventNumber")

write.csv(all_predictions_lay_dynamic, "C:/Projects/Golf/Results/Lay_Betting/Lay_Betting_FixedStakeLiability.csv", row.names = F)

# Summary
Outcome_Summary_Dynamic <- event_summaries %>%
  summarise(
    Start_Bankroll = Starting_Bankroll,
    Final_Bankroll_FS = last(Running_Bankroll_FS),
    Final_Bankroll_FL = last(Running_Bankroll_FL),
    Fixed_Stake_Profit = last(Cumulative_FS_Profit),
    Fixed_Liability_Profit = last(Cumulative_FL_Profit),
    Total_Events = n()
  )

print(Outcome_Summary_Dynamic)

#### Dynamic bankroll - Fixed Stake & Liability - Test various filters ####
all_plots <- list()

# Run analysis for model score
model_score_results <- run_filter_analysis("model_score", all_predictions_platt)
all_plots[["model_score_fixed"]] <- model_score_results$plot

# Run analysis for model rank
model_rank_results <- run_filter_analysis("model_rank", all_predictions_platt)
all_plots[["model_rank_fixed"]] <- model_rank_results$plot

# Run analysis for probability
probability_results <- run_filter_analysis("probability", all_predictions_platt)
all_plots[["probability_fixed"]] <- probability_results$plot

# Optional: Combine all results for comparison
all_filter_results <- bind_rows(
  model_score_results$data,
  model_rank_results$data,
  probability_results$data
)

# View summary of best results across all filter types
best_results_summary <- all_filter_results %>%
  group_by(Filter_Name) %>%
  summarise(
    Best_FS_Threshold = Threshold[which.max(Final_Bankroll_FS)],
    Best_FS_Bankroll = max(Final_Bankroll_FS),
    Best_FL_Threshold = Threshold[which.max(Final_Bankroll_FL)],
    Best_FL_Bankroll = max(Final_Bankroll_FL),
    .groups = 'drop'
  )

print(best_results_summary)

#### Test dynamic kelly simulation with different model score filters ####
model_score_filters <- filter_configs$model_score$thresholds

# Initialize vector to store final bankrolls and Kelly results
final_bankrolls <- numeric(length(model_score_filters))
kelly_results_model_score <- list()

# Loop through each model score filter
for (i in seq_along(model_score_filters)) {
  # Current filter value
  current_filter <- model_score_filters[i]
  
  # Create filtered dataset
  all_predictions_lay <- all_predictions_platt %>%
    select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, 
           Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
    mutate(Lay_Odds = Top20_odds * 1.1) %>%
    filter(Model_Score < current_filter)
  
  # Calculate Kelly results
  kelly_results <- calculate_kelly_lay_dynamic_bankroll(
    all_predictions_lay, 
    kelly_modifier = 1, 
    initial_bankroll = 1000, 
    max_event_exposure = 1
  )
  
  # Store Kelly results
  kelly_results_model_score[[paste0("filter_", current_filter)]] <- kelly_results
  
  # Extract final bankroll
  bankroll_history <- kelly_results$bankroll_history
  final_bankrolls[i] <- tail(bankroll_history$Ending_Bankroll, 1)
  
  # Optional: print progress
  cat("Filter:", current_filter, "- Final Bankroll:", final_bankrolls[i], "\n")
}

# Save all Kelly results for model score filters
all_kelly_results <- list()

all_kelly_results[["model_score_filters"]] <- kelly_results_model_score

# Create results dataframe
results_df <- data.frame(
  Model_Score_Filter = model_score_filters,
  Final_Bankroll = final_bankrolls,
  Return_Pct = (final_bankrolls - 1000) / 1000 * 100
)

# Create and store plot
all_plots[["kelly_model_score"]] <- create_kelly_plot(results_df, "Model Score", "Model_Score_Filter")

#### Test dynamic kelly simulation with different model rank filters ####
model_rank_filters <- filter_configs$model_rank$thresholds

# Initialize vector to store final bankrolls and Kelly results
final_bankrolls <- numeric(length(model_rank_filters))
kelly_results_model_rank <- list()

# Loop through each model rank filter
for (i in seq_along(model_rank_filters)) {
  # Current filter value
  current_filter <- model_rank_filters[i]
  
  # Create filtered dataset
  all_predictions_lay <- all_predictions_platt %>%
    select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, 
           Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
    mutate(Lay_Odds = Top20_odds * 1.1) %>%
    filter(Model_Rank < current_filter)
  
  # Calculate Kelly results
  kelly_results <- calculate_kelly_lay_dynamic_bankroll(
    all_predictions_lay, 
    kelly_modifier = 1, 
    initial_bankroll = Starting_Bankroll, 
    max_event_exposure = 1
  )
  
  # Store Kelly results
  kelly_results_model_rank[[paste0("filter_", current_filter)]] <- kelly_results
  
  # Extract final bankroll
  bankroll_history <- kelly_results$bankroll_history
  final_bankrolls[i] <- tail(bankroll_history$Ending_Bankroll, 1)
  
  # Optional: print progress
  cat("Filter:", current_filter, "- Final Bankroll:", final_bankrolls[i], "\n")
}

# Save all Kelly results for model rank filters
all_kelly_results[["model_rank_filters"]] <- kelly_results_model_rank

# Create results dataframe
results_df <- data.frame(
  Model_Rank_Filter = model_rank_filters,
  Final_Bankroll = final_bankrolls,
  Return_Pct = (final_bankrolls - 1000) / 1000 * 100
)

# Create and store plot
all_plots[["kelly_model_rank"]] <- create_kelly_plot(results_df, "Model Rank", "Model_Rank_Filter")

#### Test dynamic kelly simulation with different probability filters ####
probability_filters <- filter_configs$probability$thresholds

# Initialize vector to store final bankrolls and Kelly results
final_bankrolls <- numeric(length(probability_filters))
kelly_results_probability <- list()

# Loop through each probability filter
for (i in seq_along(probability_filters)) {
  # Current filter value
  current_filter <- probability_filters[i]
  
  # Create filtered dataset
  all_predictions_lay <- all_predictions_platt %>%
    select(EventNumber, EventID, PlayerID, Model_Score, Model_Rank, 
           Actual_Top20, Actual_Position, Top20_odds, Calibrated_Probability) %>%
    mutate(Lay_Odds = Top20_odds * 1.1) %>%
    filter(Calibrated_Probability < current_filter)
  
  # Calculate Kelly results
  kelly_results <- calculate_kelly_lay_dynamic_bankroll(
    all_predictions_lay, 
    kelly_modifier = 1, 
    initial_bankroll = Starting_Bankroll, 
    max_event_exposure = 1
  )
  
  # Store Kelly results
  kelly_results_probability[[paste0("filter_", current_filter)]] <- kelly_results
  
  # Extract final bankroll
  bankroll_history <- kelly_results$bankroll_history
  final_bankrolls[i] <- tail(bankroll_history$Ending_Bankroll, 1)
  
  # Optional: print progress
  cat("Filter:", current_filter, "- Final Bankroll:", final_bankrolls[i], "\n")
}

# Save all Kelly results for probability filters
all_kelly_results[["probability_filters"]] <- kelly_results_probability

# Create results dataframe
results_df <- data.frame(
  Probability_Filter = probability_filters,
  Final_Bankroll = final_bankrolls,
  Return_Pct = (final_bankrolls - 1000) / 1000 * 100
)

# Create and store plot
all_plots[["kelly_probability"]] <- create_kelly_plot(results_df, "Probability", "Probability_Filter")

#### DISPLAY ALL PLOTS ####
combined_plot <- grid.arrange(
  all_plots[["model_score_fixed"]],
  all_plots[["model_rank_fixed"]],
  all_plots[["probability_fixed"]],
  all_plots[["kelly_model_score"]],
  all_plots[["kelly_model_rank"]],
  all_plots[["kelly_probability"]],
  ncol = 3,
  nrow = 2
)

# Save the combined plot
ggsave("Lay_Betting_Strategy_Comparison.png", combined_plot, width = 18, height = 12, dpi = 300)

#### SAVE ALL KELLY RESULTS ####
saveRDS(all_kelly_results, "all_kelly_results.rds")

# Save individual Kelly result sets
write.csv(all_kelly_results$model_score_filters$filter_1000$results, "C:/Projects/Golf/Results/Lay_Betting/Lay_Betting_KellyCriterion.csv", row.names = F)
saveRDS(all_kelly_results$model_score_filters, "kelly_model_score_filters.rds")
saveRDS(all_kelly_results$model_rank_filters, "kelly_model_rank_filters.rds")
saveRDS(all_kelly_results$probability_filters, "kelly_probability_filters.rds")
saveRDS(Outcome_Summary_Dynamic, "No_filter_summary.rds")

#### HOW TO ACCESS SAVED KELLY RESULTS ####
cat("\n=== HOW TO ACCESS SAVED KELLY RESULTS ===\n")
cat("To reload the Kelly results in a future session:\n")
cat("all_kelly_results <- readRDS('all_kelly_results.rds')\n\n")
cat("To access specific results:\n")
cat("# Model score filter results (e.g., filter = 100):\n")
cat("model_score_100 <- all_kelly_results$model_score_filters$filter_100\n")
cat("model_score_100_bankroll <- model_score_100$bankroll_history\n")
cat("model_score_100_bets <- model_score_100$results\n\n")
cat("# Similar pattern for model_rank_filters and probability_filters\n")