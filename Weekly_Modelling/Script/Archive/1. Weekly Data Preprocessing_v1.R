#### Unified Golf Data Preprocessing Script ####
# Automatically detects and processes both historical and weekly prediction data
# Handles different data structures and column naming conventions

library(dplyr)
library(readxl)
library(openxlsx)
library(lubridate)
library(data.table)

# ===== CONFIGURATION =====
setwd("C:/Projects/Golf/")
set.seed(42)

# Input/Output file paths
WEEKLY_INPUT <- "./Weekly_Modelling/Input/This_Week_Euro.csv"
HISTORICAL_INPUT <- "./Weekly_Modelling/Input/Euro.xlsx"
WEEKLY_OUTPUT <- "./Weekly_Modelling/Input/This_Week_Processed.xlsx"
HISTORICAL_OUTPUT <- "./Weekly_Modelling/Input/PGA_Processed.xlsx"

# ===== DATA TYPE DETECTION =====
detect_data_type <- function(df) {
  # Check for key columns that distinguish dataset types
  has_eventID <- "eventID" %in% names(df)
  has_posn <- "posn" %in% names(df)
  has_date <- "Date" %in% names(df)
  has_playerID <- "playerID" %in% names(df)
  
  if (has_eventID && has_posn && has_date && has_playerID) {
    return("historical")
  } else {
    return("weekly")
  }
}

# ===== DATA LOADING AND VALIDATION =====
load_and_validate_data <- function(file_path, data_type) {
  # Load data based on file extension
  if (grepl("\\.xlsx$", file_path)) {
    df <- read_excel(file_path)
  } else {
    df <- read.csv(file_path)
  }
  
  # Fix column naming conventions
  names(df) <- gsub("^_", "X_", names(df))
  
  # Convert character columns to numeric for key variables
  numeric_cols <- c("yr3_All", "rating", "current", "X_1yr", "X_6m")
  for (col in numeric_cols) {
    if (col %in% names(df)) {
      df[[col]] <- as.numeric(df[[col]])
    }
  }
  
  # Remove rows with NA values
  df <- df[complete.cases(df),]
  
  cat("Loaded", data_type, "data with", nrow(df), "records and", ncol(df), "columns\n")
  
  return(df)
}

# ===== TARGET VARIABLE CREATION (Historical Data Only) =====
create_target_variables <- function(df) {
  if ("posn" %in% names(df)) {
    df$top_40 <- ifelse(df$posn <= 40, 1, 0)
    df$top_20 <- ifelse(df$posn <= 20, 1, 0)
    df$top_10 <- ifelse(df$posn <= 10, 1, 0)
    df$top_5 <- ifelse(df$posn <= 5, 1, 0)
    df$win <- ifelse(df$posn == 1, 1, 0)
    
    cat("Created target variables\n")
  }
  return(df)
}

# ===== BASIC FEATURE ENGINEERING =====
create_basic_features <- function(df) {
  # Performance trend calculation
  trend_cols <- c("current", "X_6m", "X_1yr", "yr3_All")
  if (all(trend_cols %in% names(df))) {
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
    cat("Created performance trend feature\n")
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
    cat("Created course advantage feature\n")
  } else {
    df$course_advantage <- NA
  }
  
  # Form indicators
  if (all(c("current_top5", "current_top20") %in% names(df))) {
    df$recent_form_ratio <- ifelse(df$current_top20 > 0, 
                                   df$current_top5 / df$current_top20, 
                                   0)
    cat("Created recent form ratio feature\n")
  }
  
  return(df)
}

# ===== EVENT-RELATIVE FEATURES =====
create_event_relative_features <- function(df, data_type) {
  if (data_type == "historical") {
    # Process by event for historical data
    events <- split(df, df$eventID)
    df_with_relatives <- data.frame()
    
    cat("Processing", length(events), "events for field-relative features\n")
    
    for (i in seq_along(events)) {
      event_data <- events[[i]]
      event_data <- process_single_event_relatives(event_data)
      df_with_relatives <- rbind(df_with_relatives, event_data)
      
      if (i %% 50 == 0) {
        cat("Processed", i, "events\n")
      }
    }
    return(df_with_relatives)
  } else {
    # Process as single event for weekly data
    return(process_single_event_relatives(df))
  }
}

process_single_event_relatives <- function(event_data) {
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
  
  return(event_data)
}

# ===== STROKES GAINED FEATURES =====
create_strokes_gained_features <- function(df, data_type) {
  # Check available SG columns
  sg_base_cols <- c("sgtee", "sgt2g", "sgapp", "sgatg", "sgp")
  sg_available <- sg_base_cols[sg_base_cols %in% names(df)]
  
  # Create combined SG metrics
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
  
  if (data_type == "historical") {
    # Process by event for historical data
    events <- split(df, df$eventID)
    df_with_sg_relatives <- data.frame()
    
    cat("Processing SG features for", length(events), "events\n")
    
    for (i in seq_along(events)) {
      event_data <- events[[i]]
      event_data <- process_single_event_sg(event_data, all_sg_columns)
      df_with_sg_relatives <- rbind(df_with_sg_relatives, event_data)
      
      if (i %% 50 == 0) {
        cat("Processed SG for", i, "events\n")
      }
    }
    return(df_with_sg_relatives)
  } else {
    # Process as single event for weekly data
    return(process_single_event_sg(df, all_sg_columns))
  }
}

process_single_event_sg <- function(event_data, all_sg_columns) {
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
  
  return(event_data)
}

# ===== MAIN PREPROCESSING FUNCTION =====
preprocess_golf_data <- function(input_file, output_file) {
  cat("\n===== PROCESSING:", input_file, "=====\n")
  
  # Step 1: Load and validate data
  df <- load_and_validate_data(input_file, "unknown")
  
  # Step 2: Detect data type
  data_type <- detect_data_type(df)
  cat("Detected data type:", data_type, "\n")
  
  # Step 3: Create target variables (historical only)
  if (data_type == "historical") {
    df <- create_target_variables(df)
  }
  
  # Step 4: Basic feature engineering
  df <- create_basic_features(df)
  
  # Step 5: Event-relative features
  df <- create_event_relative_features(df, data_type)
  
  # Step 6: Strokes gained features
  df <- create_strokes_gained_features(df, data_type)
  
  # Step 7: Save processed data
  write.xlsx(df, output_file, rowNames = FALSE)
  cat("Saved processed data to:", output_file, "\n")
  cat("Final dataset:", nrow(df), "rows,", ncol(df), "columns\n")
  
  return(df)
}

# ===== EXECUTION =====
# Check which files exist and process them
if (file.exists(WEEKLY_INPUT)) {
  cat("\nProcessing weekly prediction data...\n")
  weekly_data <- preprocess_golf_data(WEEKLY_INPUT, WEEKLY_OUTPUT)
} else {
  cat("Weekly input file not found:", WEEKLY_INPUT, "\n")
}

if (file.exists(HISTORICAL_INPUT)) {
  cat("\nProcessing historical training data...\n")
  historical_data <- preprocess_golf_data(HISTORICAL_INPUT, HISTORICAL_OUTPUT)
} else {
  cat("Historical input file not found:", HISTORICAL_INPUT, "\n")
}

cat("\n=== PREPROCESSING COMPLETE ===\n")