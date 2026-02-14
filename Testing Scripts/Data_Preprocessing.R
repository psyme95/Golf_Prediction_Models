#### Golf Betting Data Preprocessing Script ####
# Comprehensive data cleaning and feature engineering
# Run this before the main betting simulation script

library(dplyr)
library(readxl)
library(openxlsx)
library(lubridate)
library(data.table)

# ===== CONFIGURATION =====
setwd("C:/Projects/Golf")
set.seed(42)

# Input/Output file paths
INPUT_FILE <- "./Data/PGA_070725.xlsx"
QUALITY_FILE <- "./Data/FieldQuality.csv"
DATES_FILE <- "./Data/EventDates.csv"
OUTPUT_FILE <- "./Data/PGA_070725_Processed.xlsx"

# ===== DATA LOADING AND VALIDATION =====
load_and_validate_data <- function() {
  # Load main dataset
  df <- read_excel(INPUT_FILE)
  
  # Fix column naming convention
  names(df) <- gsub("^_", "X_", names(df))
  
  # Convert character columns to numeric
  df$yr3_All <- as.numeric(df$yr3_All)
  df$rating <- as.numeric(df$rating)
  
  # Remove rows with NA values
  df <- df[complete.cases(df),]
  
  return(df)
}

# ===== BASIC FEATURE ENGINEERING =====
create_basic_features <- function(df) {
  # Target variables
  df$top_40 <- ifelse(df$posn <= 40, 1, 0)
  df$top_20 <- ifelse(df$posn <= 20, 1, 0)
  df$top_10 <- ifelse(df$posn <= 10, 1, 0)
  df$top_5 <- ifelse(df$posn <= 5, 1, 0)
  df$win <- ifelse(df$posn == 1, 1, 0)
  
  # Performance trend calculation
  if (all(c("current", "X_6m", "X_1yr", "yr3_All") %in% names(df))) {
    df$performance_trend <- sapply(1:nrow(df), function(i) {
      tryCatch({
        time_points <- c(0, 1, 2, 3)
        scores <- c(df$current[i], df$X_6m[i], df$X_1yr[i], df$yr3_All[i])
        
        valid_indices <- !is.na(scores)
        if (sum(valid_indices) >= 3) {
          valid_times <- time_points[valid_indices]
          valid_scores <- scores[valid_indices]
          
          if (length(valid_scores) >= 2) {
            return(lm(valid_scores ~ valid_times)$coefficients[2])
          }
        }
        return(NA)
      }, error = function(e) NA)
    })
  } else {
    df$performance_trend <- NA
  }
  
  # Course advantage calculation
  if (all(c("course", "yr3_All") %in% names(df))) {
    df$course_advantage <- ifelse(
      !is.na(df$course) & !is.na(df$yr3_All) & df$yr3_All != 0,
      df$course / df$yr3_All,
      NA
    )
  } else {
    df$course_advantage <- NA
  }
  
  # Form indicators
  if ("current_top5" %in% names(df) && "current_top20" %in% names(df)) {
    df$recent_form_ratio <- ifelse(df$current_top20 > 0, 
                                   df$current_top5 / df$current_top20, 
                                   0)
  }

  return(df)
}

# ===== EVENT-RELATIVE FEATURES =====
create_event_relative_features <- function(df) {
  events <- split(df, df$eventID)
  df_with_relatives <- data.frame()
  
  for (i in seq_along(events)) {
    event_data <- events[[i]]
    
    # Basic rating relative features
    if ("rating" %in% names(event_data)) {
      event_mean <- mean(event_data$rating, na.rm = TRUE)
      event_median <- median(event_data$rating, na.rm = TRUE)
      event_max <- max(event_data$rating, na.rm = TRUE)
      event_min <- min(event_data$rating, na.rm = TRUE)
      event_sd <- sd(event_data$rating, na.rm = TRUE)
      
      event_data$rating_vs_field_mean <- event_data$rating - event_mean
      event_data$rating_vs_field_median <- event_data$rating - event_median
      event_data$rating_vs_field_best <- event_data$rating - event_max
      event_data$rating_vs_field_worst <- event_data$rating - event_min
      
      if (event_sd > 0) {
        event_data$rating_field_zscore <- (event_data$rating - event_mean) / event_sd
      } else {
        event_data$rating_field_zscore <- 0
      }
      
      event_data$rating_field_percentile <- rank(event_data$rating, na.last = "keep") / 
        sum(!is.na(event_data$rating))
    }
    
    # Field strength indicators
    event_data$field_size <- nrow(event_data)
    if ("rating" %in% names(event_data)) {
      event_data$field_strength <- mean(event_data$rating, na.rm = TRUE)
      event_data$field_depth <- sd(event_data$rating, na.rm = TRUE)
    }
    
    df_with_relatives <- rbind(df_with_relatives, event_data)
  }
  
  return(df_with_relatives)
}

# ===== STROKES GAINED FEATURES =====
create_strokes_gained_features <- function(df) {
  # Check available SG columns
  sg_base_cols <- c("sgtee", "sgt2g", "sgapp", "sgatg", "sgp")
  sg_available <- sg_base_cols[sg_base_cols %in% names(df)]
  
  if (all(c("sgtee", "sgapp") %in% sg_available)) {
    df$sg_ball_striking <- df$sgtee + df$sgapp
  }
  
  if (all(c("sgatg", "sgp") %in% sg_available)) {
    df$sg_short_game <- df$sgatg + df$sgp
  }
  
  # Field-relative SG features
  combined_sg_cols <- c("sg_ball_striking", "sg_short_game")
  available_combined <- combined_sg_cols[combined_sg_cols %in% names(df)]
  all_sg_columns <- c(sg_available, available_combined)
  
  events <- split(df, df$eventID)
  df_with_sg_relatives <- data.frame()
  
  for (i in seq_along(events)) {
    event_data <- events[[i]]
    
    # Process each SG metric
    for (sg_col in all_sg_columns) {
      if (sg_col %in% names(event_data)) {
        event_mean <- mean(event_data[[sg_col]], na.rm = TRUE)
        event_median <- median(event_data[[sg_col]], na.rm = TRUE)
        event_max <- max(event_data[[sg_col]], na.rm = TRUE)
        event_min <- min(event_data[[sg_col]], na.rm = TRUE)
        event_sd <- sd(event_data[[sg_col]], na.rm = TRUE)
        
        # Field-relative features
        event_data[[paste0(sg_col, "_vs_field_mean")]] <- event_data[[sg_col]] - event_mean
        event_data[[paste0(sg_col, "_vs_field_median")]] <- event_data[[sg_col]] - event_median
        event_data[[paste0(sg_col, "_vs_field_best")]] <- event_data[[sg_col]] - event_max
        
        if (event_sd > 0) {
          event_data[[paste0(sg_col, "_field_zscore")]] <- (event_data[[sg_col]] - event_mean) / event_sd
        } else {
          event_data[[paste0(sg_col, "_field_zscore")]] <- 0
        }
        
        event_data[[paste0(sg_col, "_field_percentile")]] <- rank(event_data[[sg_col]], na.last = "keep") / 
          sum(!is.na(event_data[[sg_col]]))
      }
    }
    
    df_with_sg_relatives <- rbind(df_with_sg_relatives, event_data)
  }
  
  return(df_with_sg_relatives)
}

# ===== WRAPPER FUNCTION =====
preprocess_golf_data <- function() {
  # Step 1: Load and validate data
  df <- load_and_validate_data()

  # Step 2: Basic feature engineering
  df <- create_basic_features(df)
  
  # Step 3: Event-relative features
  df <- create_event_relative_features(df)
  
  # Step 4: Strokes gained features
  df <- create_strokes_gained_features(df)
  
  # Save processed data
  write.xlsx(df, OUTPUT_FILE, rowNames = FALSE)

  return(df)
}

# ===== EXECUTION =====
# Run the preprocessing
processed_data <- preprocess_golf_data()
