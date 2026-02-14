require(dplyr)
require(biomod2)

cutoff_finder_biomod <- function(
    biomod_ensemble,  # BIOMOD.ensemble.models.out object
    biomod_data_object, # myBiomodData object
    sensitivity_target = 0.95,  # Sensitivity threshold (default 95%)
    specificity_target = 0.95   # Specificity threshold (default 95%)
) {
  # Validate input
  if(class(biomod_ensemble)[1] != "BIOMOD.ensemble.models.out") {
    stop("Input must be a BIOMOD.ensemble.models.out object")
  }
  
  if(class(biomod_data_object)[1] != "BIOMOD.formated.data"){
    stop("Must be a BIOMOD.formated.data object")
  }
  
  # Extract predictions
  predictions <- get_predictions(biomod_ensemble)$pred
  
  # Extract binary win/loss from original data
  data_species <- biomod_data_object@data.species
  
  # Create combined dataframe
  model_data <- data.frame(
    species = data_species,
    prediction = as.vector(predictions)
  )
  
  # Filter out NA values
  model_data <- model_data[!is.na(model_data$species), ]
  
  # Basic data summary
  no_of_records <- nrow(model_data)
  no_of_presence <- sum(model_data$species == 1)
  no_of_absence <- sum(model_data$species == 0)
  
  # Generate threshold sequence with 1-unit increments
  thresholds <- seq(max(model_data$prediction), 
                    min(model_data$prediction), 
                    by = -1)
  
  # Calculate sensitivity and specificity for each threshold
  performance_metrics <- lapply(thresholds, function(cutoff) {
    # Predictions at this threshold
    predicted_presences <- sum(model_data$prediction >= cutoff & model_data$species == 1)
    true_positives <- predicted_presences
    false_negatives <- no_of_presence - true_positives
    
    predicted_absences <- sum(model_data$prediction < cutoff & model_data$species == 0)
    true_negatives <- predicted_absences
    false_positives <- no_of_absence - true_negatives
    
    # Calculate metrics
    sensitivity <- (true_positives / no_of_presence) * 100
    specificity <- (true_negatives / no_of_absence) * 100
    
    # Compute TSS (True Skill Statistic)
    tss <- sensitivity + specificity - 100
    
    return(data.frame(
      threshold = cutoff,
      sensitivity = sensitivity,
      specificity = specificity,
      TSS = tss
    ))
  })
  
  # Combine results into a dataframe
  performance_df <- do.call(rbind, performance_metrics)
  
  # Find the thresholds closest to target sensitivity and specificity
  sensitivity_row <- which.min(abs(performance_df$sensitivity - (sensitivity_target * 100)))
  specificity_row <- which.min(abs(performance_df$specificity - (specificity_target * 100)))
  
  # Find the threshold with the highest TSS
  tss_row <- which.max(performance_df$TSS)
  
  # Return both the full performance dataframe and the rows closest to target thresholds
  return(list(
    full_performance = performance_df,
    target_sensitivity_threshold = performance_df[sensitivity_row, ],
    target_specificity_threshold = performance_df[specificity_row, ],
    highest_tss_threshold = performance_df[tss_row, ]
  ))
}

# Example usage:
# result <- cutoff_finder_biomod(
#    biomod_ensemble = myBiomodEM,
#    biomod_data_object = myBiomodData,
#    sensitivity_target = 0.95,
#    specificity_target = 0.95
# )