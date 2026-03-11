# ===== CONFIGURATION =====
library(biomod2)
library(dplyr)
library(readxl)
library(openxlsx)
library(lubridate)

# Set working directory
setwd("C:/Projects/Golf/Weekly_Modelling")

# Define tour configurations
TOUR_CONFIG <- list(
  PGA = list(
    name = "PGA Tour",
    models_file = "./Output/Models/PGA_Trained_Models_S26.rds",
    upcoming_file = "./Input/This_Week_PGA_Processed.xlsx"
  ),
  Euro = list(
    name = "European Tour", 
    models_file = "./Output/Models/Euro_Trained_Models_S26.rds",
    upcoming_file = "./Input/This_Week_Euro_Processed.xlsx"
  )
)

# Define base model variables (same as training script)
BASE_MODEL_VARS <- c("rating_vs_field_best",
                     "rating",
                     "rating_vs_field_worst", 
                     "yr3_All",
                     "Top5_rank",
                     "compat2",
                     "sgp_field_zscore",
                     "sgtee_field_zscore",
                     "course",
                     "sgatg_vs_field_median", 
                     "sgapp_vs_field_median",
                     "Starts_Not10",
                     "current",
                     "location", 
                     "field")

# Define betting markets configuration.
# Odds columns are intentionally excluded from all model layers —
# empirical testing showed worse outcomes when implied probability
# was included as a model feature.
BETTING_MARKETS <- list(
  "Winner" = list(target_col = "win",   odds_col = "Win_odds",  market_size = 1),
  "Top5"   = list(target_col = "top_5", odds_col = "Top5_odds", market_size = 5),
  "Top10"  = list(target_col = "top_10", odds_col = "Top10_odds", market_size = 10),
  "Top20"  = list(target_col = "top_20", odds_col = "Top20_odds", market_size = 20)
)

# ===== HELPER FUNCTIONS =====

# Apply Platt scaling to new scores using fitted parameters
predict_platt_scaling <- function(scores, platt_params) {
  1 / (1 + exp(platt_params$A * scores + platt_params$B))
}

# Get market-specific model variables.
# Odds columns are excluded from all model layers — they are only used for
# EV/edge calculations at prediction time.
get_market_model_vars <- function(market_config, base_model_vars) {
  base_vars_no_odds <- base_model_vars[!base_model_vars %in% c("Win_odds", "Top5_odds", "Top10_odds", "Top20_odds")]
  return(base_vars_no_odds)
}

# ===== PREDICTION FUNCTIONS =====

# Function to make predictions for a single model/run
predict_single_run <- function(ensemble_model, calibration_models, newdat, market_config, 
                               market_name, run_number, tour_key) {
  
  cat("    Processing Run", run_number, "\n")
  
  # Get market-specific variables
  market_model_vars <- get_market_model_vars(market_config, BASE_MODEL_VARS)
  
  # Check available variables
  available_vars <- market_model_vars[market_model_vars %in% names(newdat)]
  
  if (length(available_vars) < length(market_model_vars)) {
    missing <- market_model_vars[!market_model_vars %in% names(newdat)]
    cat("      Warning: Missing variables:", paste(missing, collapse = ", "), "\n")
  }
  
  # Prepare prediction data
  pred_env <- newdat[, available_vars]
  pred_env_xy <- newdat[, c("surname", "firstname")]
  
  # Make ensemble predictions
  proj_name <- paste0(tour_key, "_", market_name, "_Run", run_number)
  
  biomod_proj <- BIOMOD_EnsembleForecasting(
    bm.em = ensemble_model,
    proj.name = proj_name,
    new.env = pred_env,
    new.env.xy = pred_env_xy,
    models.chosen = "all",
    metric.binary = "TSS",
    metric.filter = "TSS",
    na.rm = TRUE
  )
  
  # Extract raw model scores
  model_scores <- biomod_proj@proj.out@val$pred
  
  # Create results dataframe
  results <- data.frame(
    surname = pred_env_xy$surname,
    firstname = pred_env_xy$firstname,
    model_score = model_scores,
    market_odds = newdat[[market_config$odds_col]]
  )

  # Apply Platt scaling calibration (odds excluded from all model layers)
  results$final_probability <- predict_platt_scaling(model_scores, calibration_models$platt)

  return(results)
}

# Function to process all runs for a market and average results
process_market_predictions <- function(market_name, market_training_results, 
                                       newdat, tour_key) {
  
  cat("  Processing", market_name, "market\n")
  
  market_config <- BETTING_MARKETS[[market_name]]
  run_predictions <- list()
  
  # Process each run
  for (run_idx in 1:length(market_training_results$run_results)) {
    run_data <- market_training_results$run_results[[run_idx]]
    
    predictions <- predict_single_run(
      ensemble_model = run_data$ensemble_model,
      calibration_models = run_data$calibration_models,
      newdat = newdat,
      market_config = market_config,
      market_name = market_name,
      run_number = run_idx,
      tour_key = tour_key
    )
    
    run_predictions[[run_idx]] <- predictions
  }
  
  # Average predictions across runs
  cat("    Averaging", length(run_predictions), "runs\n")
  
  # Start with first run structure
  averaged_results <- run_predictions[[1]]
  
  if (length(run_predictions) > 1) {
    # Sum up scores and probabilities
    for (run_idx in 2:length(run_predictions)) {
      averaged_results$model_score <- averaged_results$model_score + 
        run_predictions[[run_idx]]$model_score
      averaged_results$final_probability <- averaged_results$final_probability + 
        run_predictions[[run_idx]]$final_probability
      
      # Also average the other calibration methods if present
      if ("prob_glm_odds" %in% names(averaged_results)) {
        averaged_results$prob_glm_odds <- averaged_results$prob_glm_odds + 
          run_predictions[[run_idx]]$prob_glm_odds
      }
      if ("prob_glm_prob" %in% names(averaged_results)) {
        averaged_results$prob_glm_prob <- averaged_results$prob_glm_prob + 
          run_predictions[[run_idx]]$prob_glm_prob
      }
      if ("prob_platt" %in% names(averaged_results)) {
        averaged_results$prob_platt <- averaged_results$prob_platt + 
          run_predictions[[run_idx]]$prob_platt
      }
    }
    
    # Divide by number of runs to get average
    n_runs <- length(run_predictions)
    averaged_results$model_score <- averaged_results$model_score / n_runs
    averaged_results$final_probability <- averaged_results$final_probability / n_runs
    
    if ("prob_glm_odds" %in% names(averaged_results)) {
      averaged_results$prob_glm_odds <- averaged_results$prob_glm_odds / n_runs
    }
    if ("prob_glm_prob" %in% names(averaged_results)) {
      averaged_results$prob_glm_prob <- averaged_results$prob_glm_prob / n_runs
    }
    if ("prob_platt" %in% names(averaged_results)) {
      averaged_results$prob_platt <- averaged_results$prob_platt / n_runs
    }
  }
  
  # Normalise probabilities to sum to market size
  market_size <- market_config$market_size
  total_prob <- sum(averaged_results$final_probability)
  averaged_results$normalised_probability <- (averaged_results$final_probability / total_prob) * market_size
  
  # Calculate model odds
  averaged_results$model_odds <- round(1 / averaged_results$final_probability, 2)
  averaged_results$normalised_model_odds <- round(1 / averaged_results$normalised_probability, 2)
  
  # Add rating if available
  if ("rating" %in% names(newdat)) {
    averaged_results$rating <- newdat$rating[match(paste(averaged_results$surname, averaged_results$firstname),
                                                   paste(newdat$surname, newdat$firstname))]
  }
  
  # Sort by market odds
  averaged_results <- averaged_results[order(averaged_results$market_odds), ]
  
  return(averaged_results)
}

# Function to process all markets for a tour
process_tour_predictions <- function(tour_key, tour_config) {
  
  cat("\n=== PROCESSING", toupper(tour_config$name), "===\n")
  
  # Load trained models
  if (!file.exists(tour_config$models_file)) {
    cat("  Error: Models file not found:", tour_config$models_file, "\n")
    return(NULL)
  }
  
  cat("  Loading models from:", tour_config$models_file, "\n")
  trained_models <- readRDS(tour_config$models_file)
  
  # Load upcoming event data
  if (!file.exists(tour_config$upcoming_file)) {
    cat("  Error: Upcoming event file not found:", tour_config$upcoming_file, "\n")
    return(NULL)
  }
  
  cat("  Loading upcoming event from:", tour_config$upcoming_file, "\n")
  newdat <- read_excel(tour_config$upcoming_file)
  newdat <- newdat[complete.cases(newdat), ]
  cat("  Found", nrow(newdat), "players\n")
  
  # Process each market
  market_predictions <- list()
  
  for (market_name in names(trained_models)) {
    market_results <- process_market_predictions(
      market_name = market_name,
      market_training_results = trained_models[[market_name]],
      newdat = newdat,
      tour_key = tour_key
    )
    
    market_predictions[[market_name]] <- market_results
  }
  
  return(list(
    tour_name = tour_config$name,
    tour_key = tour_key,
    predictions = market_predictions,
    field_size = nrow(newdat)
  ))
}

# Function to export predictions to Excel
export_predictions <- function(tour_results, output_dir = "./Output/Predictions/") {
  
  if (is.null(tour_results)) {
    cat("  Warning: tour_results is NULL\n")
    return(NULL)
  }
  
  if (is.null(tour_results$predictions) || length(tour_results$predictions) == 0) {
    cat("  Warning: No predictions to export\n")
    return(NULL)
  }
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Create workbook
  wb <- createWorkbook()
  
  sheets_added <- 0
  
  for (market_name in names(tour_results$predictions)) {
    market_data <- tour_results$predictions[[market_name]]
    
    if (is.null(market_data) || nrow(market_data) == 0) {
      cat("    Warning: No data for", market_name, "market\n")
      next
    }
    
    # Add individual market sheet
    sheet_name <- paste0(market_name, "_Market")
    addWorksheet(wb, sheet_name)
    
    # Prepare data for export with renamed and capitalised columns
    export_data <- data.frame(
      Surname = market_data$surname,
      Firstname = market_data$firstname,
      Rating = if(!is.null(market_data$rating)) market_data$rating else NA,
      Market_Odds = market_data$market_odds,
      Model_Score = market_data$model_score,
      Probability = market_data$final_probability,  # Renamed from final_probability
      Normalised_Probability = market_data$normalised_probability,  # British spelling
      Normalised_Model_Odds = market_data$normalised_model_odds  # British spelling
    )
    
    writeData(wb, sheet_name, export_data)
    sheets_added <- sheets_added + 1
  }
  
  if (sheets_added == 0) {
    cat("  Warning: No sheets were added to workbook\n")
    return(NULL)
  }
  
  # Save workbook
  output_file <- paste0(output_dir, tour_results$tour_key, "_Predictions_", 
                        format(Sys.time(), "%d-%m-%Y"), ".xlsx")
  
  tryCatch({
    saveWorkbook(wb, output_file, overwrite = TRUE)
    cat("  Predictions saved to:", output_file, "\n")
    return(output_file)
  }, error = function(e) {
    cat("  Error saving workbook:", e$message, "\n")
    return(NULL)
  })
}

# ===== MAIN EXECUTION =====

cat("=== GOLF TOURNAMENT PREDICTIONS FROM TRAINED MODELS ===\n")

# Store results for all tours
all_tour_results <- list()
output_files <- list()

# Process each tour
for (tour_key in names(TOUR_CONFIG)) {
  tour_config <- TOUR_CONFIG[[tour_key]]
  
  tryCatch({
    # Process predictions for this tour
    tour_results <- process_tour_predictions(tour_key, tour_config)
    
    if (!is.null(tour_results)) {
      all_tour_results[[tour_key]] <- tour_results
      
      # Export to Excel
      output_file <- export_predictions(tour_results)
      if (!is.null(output_file)) {
        output_files[[tour_key]] <- output_file
      }
    }
  }, error = function(e) {
    cat("\nError processing", tour_config$name, ":", e$message, "\n")
    cat("Continuing with other tours...\n")
  })
}

# ===== FINAL SUMMARY =====

cat("\n=== PREDICTION SUMMARY ===\n")

if (length(all_tour_results) > 0) {
  for (tour_key in names(all_tour_results)) {
    tour_results <- all_tour_results[[tour_key]]
    cat("\n", tour_results$tour_name, ":\n")
    cat("  Field size:", tour_results$field_size, "players\n")
    cat("  Markets processed:", paste(names(tour_results$predictions), collapse = ", "), "\n")
    if (!is.null(output_files[[tour_key]])) {
      cat("  Output file:", basename(output_files[[tour_key]]), "\n")
    } else {
      cat("  Output file: Failed to save\n")
    }
  }
  
} else {
  cat("\nNo predictions were generated. Please check:\n")
  cat("1. Model files exist in ./Output/Models/\n")
  cat("2. Upcoming event files exist in ./Input/\n")
  cat("3. File names match the configuration\n")
}

cat("=== COMPLETE ===\n")