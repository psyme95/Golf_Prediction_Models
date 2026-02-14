#preds <- unnest(sorted_predictions, "players")

preds <- all_predictions %>%
  select(EventID, PlayerID, Actual_Position, Model_Score, Model_Rank, Top20_Winnings, Top20_Odds, Rating)

bin_width <- 100 

preds_binned <- preds %>%
  mutate(
    score_bin = floor(Model_Score / bin_width) * bin_width,
    score_bin_center = score_bin + bin_width/2
  ) %>%
  group_by(score_bin, score_bin_center) %>%
  summarise(
    total_players = n(),
    top20_players = sum(Actual_Position <= 20, na.rm = TRUE),
    top20_proportion = top20_players / total_players,
    .groups = 'drop'
  )

preds <- preds %>%
  arrange(EventID,
          desc(Model_Score),
          desc(Rating)) %>%
  mutate(
    score_bin = floor(Model_Score / bin_width) * bin_width,
    score_bin_floor = score_bin,
    score_bin_ceiling = score_bin + bin_width - 1
  ) %>%
  left_join(preds_binned %>% select(score_bin, top20_proportion), 
            by = "score_bin") %>%
  mutate(min_odds = 1 / top20_proportion,
         good_odds = Top20_Odds >= (min_odds + 1),
         profit = case_when(
           good_odds == TRUE & Actual_Position <= 20 ~ Top20_Winnings - 10,
           good_odds == TRUE & Actual_Position > 20 ~ -10,
           good_odds == FALSE ~ 0,
           TRUE ~ 0
         ))

sum(preds$profit, na.rm = TRUE)
sd(preds[preds$good_odds == TRUE, "profit"])
mean(preds[preds$good_odds == TRUE, "profit"])
