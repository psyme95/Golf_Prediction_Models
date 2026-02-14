library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(scales)

df <- read_excel("C:/Projects/Golf/Betting Simulations/Top 40/PGA_Top40_LOOCV_Results_Run2d.xlsx")

# First, get unique EventID-Quality pairs from df_with_diffs
quality_lookup <- df_with_diffs %>%
  select(eventID, Quality) %>%
  distinct()

field_lookup <- df_with_diffs %>%
  select(eventID, field) %>%
  distinct()

# Now join - this should preserve your original row count
df <- df %>%
  left_join(quality_lookup, by = c("EventID" = "eventID")) %>%
  left_join(field_lookup, by = c("EventID" = "eventID")) %>%
  filter(Num_Bets > 0) %>%
  mutate(
    Quality_Bin = floor(Quality * 2) / 2,  # Rounds down to nearest 0.5
    Quality_Range = paste0(Quality_Bin, "-", Quality_Bin + 0.5),
    Range_Label = paste0(Range_Start, "-", Range_End)
  )


# Analyze performance by quality bin and betting range
performance_summary <- df %>%
  filter(Num_Bets > 0) %>%  # Only include ranges with actual bets
  group_by(Quality_Range, Range_Label) %>%
  #filter(Num_Bets > 0 & Quality > 71) %>%  # Only include ranges with actual bets
  group_by(Range_Label) %>%
  summarise(
    Avg_ROI = mean(ROI, na.rm = TRUE),
    ROI_SD = sd(ROI, na.rm = TRUE),
    Sharpe_Ratio = mean(ROI, na.rm = T) / sd(ROI, na.rm = T),
    Total_Bets = sum(Num_Bets),
    Total_Profit = sum(Profit),
    Success_Rate = mean(Success_Rate, na.rm = TRUE),
    Event_Count = n_distinct(EventID),
    .groups = "drop"
  ) %>%
  #arrange(desc(Sharpe_Ratio))
  arrange(Quality_Range, desc(Sharpe_Ratio))

# View the results
print(performance_summary)

# Get the best performing range for each quality bin
best_ranges <- performance_summary %>%
  group_by(Quality_Range) %>%
  slice_max(Total_Profit, n = 2) %>%
  ungroup()

print("Best performing ranges by quality:")
print(best_ranges)


# Heatmap showing ROI by quality range and betting range
ggplot(performance_summary, aes(x = Range_Label, y = Quality_Range, fill = Avg_ROI)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", mid = "white", high = "green", midpoint = 0) +
  labs(title = "Average ROI by Quality Range and Betting Range",
       x = "Betting Range",
       y = "Quality Range",
       fill = "Avg ROI (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

write.xlsx(performance_summary, "C:/Projects/Golf/Betting Simulations/Top 40/Top40_Run2d_PerformanceSummary.xlsx", row.names=F, quote=T)



# Check Failed Events
Events <- read_excel("C:/Projects/Golf/Betting Simulations/Top 40/PGA_Top40_LOOCV_Results_Run2d.xlsx", sheet="Processing_Summary")

Events <- Events %>%
  filter(status != "completed") %>%
  left_join(quality_lookup, by = "eventID") %>%
  left_join(field_lookup, by = "eventID")


# QUALITY SCORE ANALYSIS
quality_stats <- Events %>%
  group_by(status) %>%
  summarise(
    min_quality = min(Quality, na.rm = TRUE),
    q25_quality = quantile(Quality, 0.25, na.rm = TRUE),
    median_quality = median(Quality, na.rm = TRUE),
    q75_quality = quantile(Quality, 0.75, na.rm = TRUE),
    max_quality = max(Quality, na.rm = TRUE),
    sd_quality = round(sd(Quality, na.rm = TRUE), 2)
  )

print(quality_stats)
