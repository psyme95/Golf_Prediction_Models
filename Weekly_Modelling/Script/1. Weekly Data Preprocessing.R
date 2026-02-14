#### Unified Golf Data Preprocessing Script - PGA & European Tours ####
# Processes both historical and weekly prediction data for PGA and European tours
# Handles separate files for each tour type

library(dplyr)
library(readxl)
library(openxlsx)
library(lubridate)
library(data.table)

# ===== CONFIGURATION =====
setwd("C:/Projects/Golf/")
set.seed(42)

# Tour configurations
TOURS <- list(
  pga = list(
    name = "PGA Tour",
    historical_input = "./Weekly_Modelling/Input/PGA.xlsx",
    weekly_input = "./Weekly_Modelling/Input/This_Week_PGA.csv",
    historical_output = "./Weekly_Modelling/Input/PGA_Processed.xlsx",
    weekly_output = "./Weekly_Modelling/Input/This_Week_PGA_Processed.xlsx"
  ),
  euro = list(
    name = "European Tour",
    historical_input = "./Weekly_Modelling/Input/Euro.xlsx",
    weekly_input = "./Weekly_Modelling/Input/This_Week_Euro.csv",
    historical_output = "./Weekly_Modelling/Input/Euro_Processed.xlsx",
    weekly_output = "./Weekly_Modelling/Input/This_Week_Euro_Processed.xlsx"
  )
)

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
  
  # Remove problematic columns with lots of missing data (historical only)
  historical_only_cols <- c("Lay_odds", "Top40_odds", "EW_Profit", "Top5_Profit", 
                            "Top10_Profit", "Top20_Profit", "Top40_Profit", 
                            "Lay_top5", "Lay_top10", "Lay_top20", "Rd2Pos",	"Rd2Lead", "Betfair_rd2")
  
  # Check if this looks like historical data and remove these columns
  if (data_type == "historical" || any(historical_only_cols %in% names(df))) {
    cols_to_remove <- historical_only_cols[historical_only_cols %in% names(df)]
    if (length(cols_to_remove) > 0) {
      cat("Removing", length(cols_to_remove), "historical-only columns with missing data:", 
          paste(cols_to_remove, collapse = ", "), "\n")
      df <- select(df, -all_of(cols_to_remove))
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
preprocess_golf_data <- function(input_file, output_file, tour_name) {
  cat("\n===== PROCESSING:", tour_name, "-", input_file, "=====\n")
  
  # Step 1: Load and validate data
  df <- load_and_validate_data(input_file, "unknown")
  
  # Step 2: Detect data type
  data_type <- detect_data_type(df)
  cat("Detected data type:", data_type, "\n")
  
  # Step 3: Create target variables (historical only)
  if (data_type == "historical") {
    df <- create_target_variables(df)
  }
  
  # Step 4: Event-relative features
  df <- create_event_relative_features(df, data_type)
  
  # Step 5: Strokes gained features
  df <- create_strokes_gained_features(df, data_type)
  
  # Step 6: Save processed data
  write.xlsx(df, output_file, rowNames = FALSE)
  cat("Saved processed data to:", output_file, "\n")
  cat("Final dataset:", nrow(df), "rows,", ncol(df), "columns\n")
  
  return(df)
}

# ===== TOUR PROCESSING FUNCTION =====
process_tour <- function(tour_config, tour_key) {
  tour_name <- tour_config$name
  results <- list()
  
  cat("PROCESSING", toupper(tour_name), "\n")

  # Process historical data
  if (file.exists(tour_config$historical_input)) {
    cat("\nProcessing", tour_name, "historical data...\n")
    results$historical <- preprocess_golf_data(
      input_file = tour_config$historical_input,
      output_file = tour_config$historical_output,
      tour_name = paste(tour_name, "Historical")
    )
  } else {
    cat("Historical file not found:", tour_config$historical_input, "\n")
  }
  
  # Process weekly data
  if (file.exists(tour_config$weekly_input)) {
    cat("\nProcessing", tour_name, "weekly prediction data...\n")
    results$weekly <- preprocess_golf_data(
      input_file = tour_config$weekly_input,
      output_file = tour_config$weekly_output,
      tour_name = paste(tour_name, "Weekly")
    )
  } else {
    cat("Weekly file not found:", tour_config$weekly_input, "\n")
  }
  
  return(results)
}

# ===== EXECUTION =====
cat("=== GOLF DATA PREPROCESSING - PGA & EUROPEAN TOURS ===\n")
cat("Current working directory:", getwd(), "\n")

# Initialize results storage
all_results <- list()

# Process each tour
for (tour_key in names(TOURS)) {
  tour_results <- process_tour(TOURS[[tour_key]], tour_key)
  all_results[[tour_key]] <- tour_results
}

# ===== SUMMARY =====
cat("PREPROCESSING COMPLETE - SUMMARY\n")

for (tour_key in names(all_results)) {
  tour_name <- TOURS[[tour_key]]$name
  cat("\n", tour_name, ":\n")
  
  if (!is.null(all_results[[tour_key]]$historical)) {
    hist_rows <- nrow(all_results[[tour_key]]$historical)
    hist_cols <- ncol(all_results[[tour_key]]$historical)
    cat("  Historical: ", hist_rows, " records, ", hist_cols, " features\n")
  } else {
    cat("  Historical: Not processed\n")
  }
  
  if (!is.null(all_results[[tour_key]]$weekly)) {
    weekly_rows <- nrow(all_results[[tour_key]]$weekly)
    weekly_cols <- ncol(all_results[[tour_key]]$weekly)
    cat("  Weekly: ", weekly_rows, " players, ", weekly_cols, " features\n")
  } else {
    cat("  Weekly: Not processed\n")
  }
}

cat("\nOutput files saved to ./Weekly_Modelling/Input/\n")
cat("Ready for modeling!\n")