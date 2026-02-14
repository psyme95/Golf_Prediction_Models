field_size <- new_preds %>%
  group_by(EventID) %>%
  summarise(field = n())

event_stake <- kelly_sim$bet_history %>%
  group_by(EventID) %>%
  summarise(event_stake = sum(bet_amount))

kelly_sim$bankroll_history <- kelly_sim$bankroll_history %>%
  left_join(field_size, by = "EventID") %>%
  left_join(event_stake, by = "EventID") %>%
  left_join(eventDates, by = "EventID") %>%
  left_join(eventQuality, by = "EventID") %>%
  mutate(event_profit = cumulative_profit - lag(cumulative_profit, default = 0),
         profit_pct = (event_profit / event_stake) * 100)

plot(kelly_sim$bankroll_history$field, kelly_sim$bankroll_history$profit_pct)
abline(lm(kelly_sim$bankroll_history$profit_pct ~ kelly_sim$bankroll_history$field))

fit <- lm(kelly_sim$bankroll_history$profit_pct ~ kelly_sim$bankroll_history$field)
summary(fit)


plot(kelly_sim$bankroll_history$event_number, kelly_sim$bankroll_history$profit_pct)
abline(lm(kelly_sim$bankroll_history$profit_pct ~ kelly_sim$bankroll_history$event_number))

fit <- lm(kelly_sim$bankroll_history$profit_pct ~ kelly_sim$bankroll_history$event_number)
summary(fit)

plot(kelly_sim$bankroll_history$Quality, kelly_sim$bankroll_history$profit_pct)
abline(lm(kelly_sim$bankroll_history$profit_pct ~ kelly_sim$bankroll_history$Quality))

fit <- lm(kelly_sim$bankroll_history$profit_pct ~ kelly_sim$bankroll_history$Quality)
summary(fit)
