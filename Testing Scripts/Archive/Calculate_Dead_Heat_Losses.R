library(readxl)
library(dplyr)
library(stringr)

summary <- read_excel("./Betting Simulations/8020_Sequential_Analysis_0611_0910.xlsx", sheet="Summary")
event_by_event <- read_excel("./Betting Simulations/8020_Sequential_Analysis_0611_0910.xlsx", sheet="Event_By_Event")
player_predictions <- read_excel("./Betting Simulations/8020_Sequential_Analysis_0611_0910.xlsx", sheet="Player_Predictions")

player_predictions <- player_predictions %>%
  mutate(dead_heat = ifelse(10*top40_odds > top40_winnings+0.01 & actual_position <= 40, "Y", "N"),
         dead_heat_loss = ifelse(dead_heat == "Y" & prediction_correct == 1, top40_odds*10 - top40_winnings, 0)) 

# Function to extract rank range from strategy string
extract_rank_range <- function(strategy) {
  # Extract the two numbers from patterns like "R1to10", "R11to20", etc.
  numbers <- as.numeric(str_extract_all(strategy, "\\d+")[[1]])
  return(numbers[1]:numbers[2])
}

# Create a lookup table for all combinations
event_strategy_lookup <- event_by_event %>%
  select(eventID, strategy) %>%
  rowwise() %>%
  mutate(
    rank_range = list(extract_rank_range(strategy))
  ) %>%
  ungroup() %>%
  unnest_longer(rank_range) %>%
  rename(model_rank = rank_range)

# Calculate dead heat losses
dead_heat_sums <- player_predictions %>%
  inner_join(event_strategy_lookup, by = c("eventID", "model_rank")) %>%
  group_by(eventID, strategy) %>%
  summarise(dead_heat_loss_sum = sum(dead_heat_loss, na.rm = TRUE), .groups = "drop")

# Join back to original data
event_by_event <- event_by_event %>%
  left_join(dead_heat_sums, by = c("eventID", "strategy"))

# Calculate Stategy Dead Heat Losses
strategy_DHL <- event_by_event %>%
  group_by(strategy) %>%
  summarise(strategy_DHL = sum(dead_heat_loss_sum))

summary <- summary %>%
  left_join(strategy_DHL, by = "strategy") %>%
  mutate(profit_without_dead_heat = total_profit + strategy_DHL)

# Write updates csvs
write.csv(summary, "./Betting Simulations/summary.csv", row.names=F)
write.csv(event_by_event, "./Betting Simulations/event_by_event.csv", row.names=F)
write.csv(player_predictions, "./Betting Simulations/player_predictions.csv", row.names=F)
