#### Load Packages ####
library(dplyr)
library(ggplot2)
library(tidyr)
library(caret)
library(glmnet)
library(randomForest)

#### Define Feature Selection Function ####
run_feature_selection <- function(expl_vars, resp_var, n_features=15, use_lasso=TRUE, seed=123) {
  require(caret)
  require(glmnet)
  require(randomForest)
  
  set.seed(seed)
  
  cat("\n---------------------------------------------\n")
  cat("STARTING AUTOMATED FEATURE SELECTION PIPELINE\n")
  cat("---------------------------------------------\n")
  cat(sprintf("Using seed: %d\n", seed))
  
  start_time <- Sys.time()
  cat(sprintf("Start time: %s\n\n", start_time))
  
  # 1. Remove highly correlated features
  cat("STEP 1: Removing highly correlated features...\n")
  cat(sprintf("- Initial number of variables: %d\n", ncol(expl_vars)))
  cor_matrix <- cor(expl_vars, use="pairwise.complete.obs")
  non_correlated_idx <- findCorrelation(cor_matrix, cutoff=0.75, names=FALSE)
  
  if(length(non_correlated_idx) > 0) {
    reduced_vars <- expl_vars[, -non_correlated_idx]
    cat(sprintf("- Removed %d highly correlated variables\n", length(non_correlated_idx)))
  } else {
    reduced_vars <- expl_vars
    cat("- No highly correlated variables found\n")
  }
  cat(sprintf("- Remaining variables: %d\n", ncol(reduced_vars)))
  cat(sprintf("- Step 1 completed in %.2f seconds\n\n", 
              as.numeric(difftime(Sys.time(), start_time, units="secs"))))
  
  # 2. Run initial importance screening with Random Forest
  step2_time <- Sys.time()
  cat("STEP 2: Running initial Random Forest for variable importance...\n")
  cat("- Training Random Forest model (this may take a while)...\n")
  rf_model <- randomForest(x=reduced_vars, y=as.factor(resp_var), 
                           importance=TRUE, ntree=200)
  var_imp <- importance(rf_model)
  sorted_imp <- sort(var_imp[,"MeanDecreaseGini"], decreasing=TRUE)
  
  # Display top 10 variables by importance
  cat("- Top 10 variables by importance:\n")
  top10 <- head(sorted_imp, 10)
  for(i in 1:length(top10)) {
    cat(sprintf("  %2d. %-30s: %.4f\n", i, names(top10)[i], top10[i]))
  }
  
  # Select candidates for next stage
  candidate_count <- min(n_features*2, length(sorted_imp))
  candidate_features <- names(sorted_imp)[1:candidate_count]
  cat(sprintf("- Selected top %d variables as candidates for further selection\n", candidate_count))
  cat(sprintf("- Step 2 completed in %.2f seconds\n\n", 
              as.numeric(difftime(Sys.time(), step2_time, units="secs"))))
  
  # Store LASSO results
  lasso_results <- NULL
  lasso_features <- NULL
  
  # 3. Apply Relaxed LASSO when requested
  if (use_lasso) {
    step3_time <- Sys.time()
    cat("STEP 3: Applying Relaxed LASSO for feature selection...\n")
    
    # Prepare X and y for glmnet
    X <- as.matrix(reduced_vars[,candidate_features])
    
    # For classification
    if (is.factor(resp_var) || length(unique(resp_var)) <= 10) {
      y <- as.factor(resp_var)
      family <- "binomial"
      cat("- Using logistic LASSO for classification task\n")
    } else {
      # For regression
      y <- resp_var
      family <- "gaussian"
      cat("- Using linear LASSO for regression task\n")
    }
    
    # Stage 1: Standard LASSO for variable selection
    cat("- Running cross-validation to find optimal lambda...\n")
    cv_lasso <- cv.glmnet(X, y, family=family, alpha=1, nfolds=3)
    
    # Get coefficients at lambda.min (less aggressive than lambda.1se)
    lasso_coef <- coef(cv_lasso, s="lambda.min")
    
    # Get selected features (excluding intercept)
    selected_idx <- which(lasso_coef[-1] != 0)
    if (length(selected_idx) > 0) {
      lasso_features <- candidate_features[selected_idx]
      cat(sprintf("- LASSO selected %d features\n", length(lasso_features)))
    } else {
      # If LASSO didn't select any features, use top 5 from RF
      lasso_features <- candidate_features[1:min(5, length(candidate_features))]
      cat("- LASSO didn't select any features, using top 5 from Random Forest\n")
    }
    
    # Make sure we have enough features for RFE
    if (length(lasso_features) < 5 && length(candidate_features) >= 5) {
      # Add more features from RF ranking
      additional_needed <- 5 - length(lasso_features)
      additional_features <- setdiff(candidate_features[1:10], lasso_features)[1:additional_needed]
      if (length(additional_features) > 0) {
        lasso_features <- c(lasso_features, additional_features)
        cat(sprintf("- Added %d features from RF ranking to ensure enough features for RFE\n", 
                    length(additional_features)))
      }
    }
    
    # Display the LASSO selected features
    cat("- LASSO selected features:\n")
    for(i in 1:length(lasso_features)) {
      cat(sprintf("  %2d. %s\n", i, lasso_features[i]))
    }
    
    # Store LASSO results for later
    lasso_results <- list(
      lambda = cv_lasso$lambda.min,
      selected_features = lasso_features
    )
    
    # Use LASSO features for RFE
    candidate_features <- lasso_features
    
    cat(sprintf("- Step 3 completed in %.2f seconds\n\n", 
                as.numeric(difftime(Sys.time(), step3_time, units="secs"))))
  } else {
    cat("STEP 3: Skipping LASSO (use_lasso=FALSE)\n\n")
  }
  
  # 4. Run RFE on these candidates
  step4_time <- Sys.time()
  cat(sprintf("STEP %d: Running Recursive Feature Elimination...\n", 
              ifelse(use_lasso, 4, 3)))
  
  # Check if we have enough candidates for RFE
  if (length(candidate_features) < 5) {
    cat(sprintf("- Only %d candidate features available, skipping RFE\n", length(candidate_features)))
    final_features <- candidate_features
  } else {
    cat(sprintf("- Running RFE with %d-fold cross-validation\n", 5))
    
    # Set up sizes to test - make sure it's a valid sequence
    max_size <- min(n_features, length(candidate_features))
    min_size <- min(5, max_size)
    
    if (max_size < 1) {
      cat("- No features available for RFE, using LASSO features directly\n")
      final_features <- lasso_features
    } else {
      if (min_size == max_size) {
        sizes <- min_size
        cat(sprintf("- Only %d candidate features available, using all of them\n", min_size))
      } else {
        sizes <- seq(min_size, max_size, by=1)
        cat(sprintf("- Testing feature subset sizes: %s\n", paste(sizes, collapse=", ")))
      }
      
      # Create a simplified control object with NA for seeds
      ctrl <- rfeControl(functions=rfFuncs, 
                         method="cv", 
                         number=2,
                         returnResamp="final",
                         verbose=TRUE,
                         seeds=NA,
                         allowParallel=FALSE)
      
      cat("- Starting RFE process:\n")
      
      # Verify that the columns exist
      valid_columns <- candidate_features %in% colnames(reduced_vars)
      if (!all(valid_columns)) {
        missing_columns <- candidate_features[!valid_columns]
        cat(sprintf("WARNING: Missing columns in data: %s\n", 
                    paste(missing_columns, collapse=", ")))
        candidate_features <- candidate_features[valid_columns]
        cat(sprintf("- Proceeding with %d valid features\n", length(candidate_features)))
      }
      
      # Double check we have data to work with
      if (length(candidate_features) == 0) {
        cat("ERROR: No valid features available for RFE\n")
        # Fall back to RF ranking
        final_features <- names(sorted_imp)[1:min(n_features, length(sorted_imp))]
      } else {
        # Create data subset for RFE
        rfe_data <- reduced_vars[, candidate_features, drop=FALSE]
        
        # Try RFE but handle errors
        result <- try({
          results <- rfe(x=rfe_data, 
                         y=as.factor(resp_var),
                         sizes=sizes,
                         rfeControl=ctrl,
                         preProcess=c("center", "scale"),
                         metric="Accuracy")
          
          # Show optimal variables
          cat(sprintf("- Optimal number of features: %d\n", results$optsize))
          cat("- Final selected features:\n")
          final_features <- predictors(results)
          for(i in 1:length(final_features)) {
            cat(sprintf("  %2d. %s\n", i, final_features[i]))
          }
          
          # Show performance metrics at optimal size
          optimal_idx <- which(results$results$Variables == results$optsize)
          cat(sprintf("- Cross-validation accuracy: %.4f\n", 
                      results$results$Accuracy[optimal_idx]))
        }, silent=FALSE)
        
        # Check if RFE failed
        if (inherits(result, "try-error")) {
          cat("ERROR encountered during RFE. Falling back to alternative selection.\n")
          
          # Provide alternative based on RF importance ranking or LASSO results
          if (use_lasso && !is.null(lasso_features) && length(lasso_features) > 0) {
            cat("Using LASSO-selected features:\n")
            final_features <- lasso_features[1:min(n_features, length(lasso_features))]
          } else {
            cat("Using Random Forest importance ranking:\n")
            final_features <- names(sorted_imp)[1:min(n_features, length(sorted_imp))]
          }
          
          for(i in 1:length(final_features)) {
            cat(sprintf("  %2d. %s\n", i, final_features[i]))
          }
        }
      }
    }
  }
  
  # Calculate final execution time
  end_time <- Sys.time()
  total_time <- difftime(end_time, start_time, units="mins")
  cat(sprintf("\nFeature selection completed in %.2f minutes\n", 
              as.numeric(total_time)))
  cat("---------------------------------------------\n")
  
  return(list(
    selected_features = final_features,
    rfe_results = if(exists("results")) results else NULL,
    execution_time = total_time,
    importance = data.frame(
      feature = names(sorted_imp),
      importance = sorted_imp,
      stringsAsFactors = FALSE
    ),
    lasso_results = lasso_results
  ))
}

#### Define Multi-Seed Feature Selection Function ####
run_multi_seed_feature_selection <- function(expl_vars, resp_var, n_features=15, use_lasso=TRUE, 
                                             num_runs=10, seed_range=c(100, 1000)) {
  
  cat("\n=========================================================\n")
  cat("RUNNING FEATURE SELECTION WITH MULTIPLE SEEDS\n")
  cat("=========================================================\n")
  
  # Generate random seeds
  set.seed(123)  # For reproducible seed generation
  seeds <- sample(seed_range[1]:seed_range[2], num_runs, replace=FALSE)
  cat(sprintf("Generated %d random seeds: %s\n\n", 
              num_runs, paste(seeds, collapse=", ")))
  
  # Storage for results
  all_results <- list()
  all_importance_scores <- list()
  selected_features_all <- list()
  
  # Run feature selection multiple times
  for (i in 1:num_runs) {
    cat(sprintf("\n\n=========== RUN %d/%d (SEED: %d) ===========\n\n", 
                i, num_runs, seeds[i]))
    
    # Run feature selection with current seed
    result <- try({
      results <- run_feature_selection(expl_vars, resp_var, n_features, use_lasso, seed=seeds[i])
      
      # Store results
      all_results[[i]] <- results
      all_importance_scores[[i]] <- results$importance
      
      # Store selected features for this run
      selected_features_all[[i]] <- results$selected_features
      
      # Report success
      cat(sprintf("\nRun %d completed successfully with seed %d\n", i, seeds[i]))
    }, silent=FALSE)
    
    # Check if run failed
    if (inherits(result, "try-error")) {
      cat(sprintf("\nERROR in run %d with seed %d\n", i, seeds[i]))
      # Store empty results for this failed run
      all_results[[i]] <- NULL
      all_importance_scores[[i]] <- NULL
      selected_features_all[[i]] <- NULL
    }
  }
  
  # Remove NULL entries from any failed runs
  all_results <- all_results[!sapply(all_results, is.null)]
  all_importance_scores <- all_importance_scores[!sapply(all_importance_scores, is.null)]
  selected_features_all <- selected_features_all[!sapply(selected_features_all, is.null)]
  
  # Calculate successful runs
  successful_runs <- length(all_results)
  if (successful_runs == 0) {
    stop("All feature selection runs failed. Please check your data and parameters.")
  }
  
  cat(sprintf("\n%d of %d runs completed successfully\n", successful_runs, num_runs))
  
  # Count feature selection frequency
  all_selected_features <- unlist(selected_features_all)
  feature_counts <- table(all_selected_features)
  selected_features_count <- as.list(feature_counts)
  
  # Combine importance scores from all runs
  all_imp_df <- bind_rows(lapply(seq_along(all_importance_scores), function(i) {
    df <- all_importance_scores[[i]]
    df$run <- i
    df$seed <- seeds[i]
    return(df)
  }))
  
  # Calculate average importance scores
  avg_importance <- all_imp_df %>%
    group_by(feature) %>%
    summarize(
      avg_importance = mean(importance, na.rm = TRUE),
      sd_importance = sd(importance, na.rm = TRUE),
      min_importance = min(importance, na.rm = TRUE),
      max_importance = max(importance, na.rm = TRUE),
      num_appearances = n()
    ) %>%
    arrange(desc(avg_importance))
  
  # Convert feature selection counts to data frame
  selection_freq <- data.frame(
    feature = names(selected_features_count),
    selection_count = as.numeric(selected_features_count),
    selection_percentage = 100 * as.numeric(selected_features_count) / successful_runs
  ) %>%
    arrange(desc(selection_count))
  
  # Join average importance with selection frequency
  feature_stats <- full_join(avg_importance, selection_freq, by = "feature") %>%
    mutate(
      selection_count = ifelse(is.na(selection_count), 0, selection_count),
      selection_percentage = ifelse(is.na(selection_percentage), 0, selection_percentage)
    ) %>%
    arrange(desc(avg_importance))
  
  # Print summary
  cat("\n=========================================================\n")
  cat("FEATURE SELECTION SUMMARY ACROSS ALL RUNS\n")
  cat("=========================================================\n")
  
  cat(sprintf("\nFeatures selected by frequency (out of %d successful runs):\n", successful_runs))
  top_by_freq <- head(selection_freq, 20)
  for(i in 1:nrow(top_by_freq)) {
    cat(sprintf("%2d. %-30s: %2d runs (%.1f%%)\n", 
                i, top_by_freq$feature[i], 
                top_by_freq$selection_count[i], 
                top_by_freq$selection_percentage[i]))
  }
  
  cat("\nTop 20 features by average importance score:\n")
  top_by_imp <- head(feature_stats, 20)
  for(i in 1:nrow(top_by_imp)) {
    cat(sprintf("%2d. %-30s: %.4f (SD: %.4f, Selected in %d/%d runs)\n", 
                i, top_by_imp$feature[i], 
                top_by_imp$avg_importance[i], 
                top_by_imp$sd_importance[i],
                ifelse(is.na(top_by_imp$selection_count[i]), 0, top_by_imp$selection_count[i]), 
                successful_runs))
  }
  
  # Return comprehensive results
  return(list(
    avg_importance = avg_importance,
    selection_frequency = selection_freq,
    feature_stats = feature_stats,
    all_runs = all_results,
    all_importance_scores = all_imp_df,
    seeds = seeds,
    successful_runs = successful_runs,
    total_runs = num_runs
  ))
}

#### Load Data ####
setwd("A:/OneDrive - University of Southampton/Golf")
df <- read.csv("A:/OneDrive - University of Southampton/Golf/Data/PGA_withodds.csv")

#### Sample Run Code ####
df$rating <- as.numeric(df$rating)
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
df <- df[complete.cases(df),]
eventQuality <- read.csv("A:/OneDrive - University of Southampton/Golf/Data/FieldQuality.csv")

df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID")) %>%
  filter(Quality <= 71)

pgalist <- split(df, df$eventID)

# Add more stats for each tournament
for (eventID in names(pgalist)) {
  pgalist[[eventID]]$rating_normal <- scale(pgalist[[eventID]]$rating)
  pgalist[[eventID]]$rating_mean <- mean(pgalist[[eventID]]$rating, na.rm = TRUE)
  pgalist[[eventID]]$rating_median <- median(pgalist[[eventID]]$rating, na.rm = TRUE)
  pgalist[[eventID]]$diff_from_mean <- pgalist[[eventID]]$rating - mean(pgalist[[eventID]]$rating, na.rm = TRUE)
  pgalist[[eventID]]$diff_from_median <- pgalist[[eventID]]$rating - median(pgalist[[eventID]]$rating, na.rm = TRUE)
  pgalist[[eventID]]$diff_from_max <- pgalist[[eventID]]$rating - max(pgalist[[eventID]]$rating, na.rm = TRUE)
  pgalist[[eventID]]$diff_from_min <- pgalist[[eventID]]$rating - min(pgalist[[eventID]]$rating, na.rm = TRUE)
  pgalist[[eventID]]$top_40 <- ifelse(pgalist[[eventID]]$posn <= 40, 1, 0)
  pgalist[[eventID]]$rating_normal <- scale(pgalist[[eventID]]$rating)
  pgalist[[eventID]]$log_rating <- log(pgalist[[eventID]]$rating+50)
  pgalist[[eventID]]$diff_from_field_quality <-pgalist[[eventID]]$rating - pgalist[[eventID]]$Quality
}

pgalist_filtered <- pgalist[sapply(pgalist, function(df) nrow(df) > 40)]

# Combine all tournament data into one dataframe
all_players_data <- data.frame()

for (eventID in names(pgalist_filtered)) {
  temp_df <- pgalist_filtered[[eventID]]
  temp_df$eventID <- eventID
  all_players_data <- rbind(all_players_data, temp_df)
}

resp <- all_players_data[, "top_40"]
expl <- all_players_data[, !names(all_players_data) %in% c("eventID", "posn", "playerID","score", "win", "top_40")]
expl <- sapply(expl, as.numeric)

# Run multi-seed feature selection
multi_seed_results <- run_multi_seed_feature_selection(
  expl_vars = expl, 
  resp_var = resp, 
  n_features = 20, 
  use_lasso = TRUE,
  num_runs = 10,  # Number of times to run with different seeds
  seed_range = c(1000, 9999)  # Range for random seed generation
)

# Save results
saveRDS(multi_seed_results, file = "multi_seed_feature_selection_results.rds")

# Create plots
# Plot average importance
top_features <- head(multi_seed_results$feature_stats, 20)

# Plot of average importance with error bars
ggplot(top_features, aes(x = reorder(feature, avg_importance), y = avg_importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_errorbar(aes(ymin = avg_importance - sd_importance, 
                    ymax = avg_importance + sd_importance), 
                width = 0.2) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 20 Features by Average Importance",
    subtitle = paste("Across", multi_seed_results$seeds %>% length(), "runs with different seeds"),
    x = "Feature",
    y = "Average Importance Score (± SD)"
  )

# Plot of selection frequency
ggplot(top_features, aes(x = reorder(feature, selection_percentage), y = selection_percentage)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Feature Selection Frequency",
    subtitle = paste("Percentage of runs where feature was selected (out of", 
                     multi_seed_results$seeds %>% length(), "runs)"),
    x = "Feature",
    y = "Selection Percentage (%)"
  )

# Plot of importance across different runs
# Get top 10 features by average importance
top10_features <- head(multi_seed_results$avg_importance$feature, 20)

# Filter importance scores for these features
top10_scores <- multi_seed_results$all_importance_scores %>%
  filter(feature %in% top10_features)

# Create boxplot of importance across runs
ggplot(top10_scores, aes(x = reorder(feature, importance, FUN = median), y = importance)) +
  geom_boxplot(fill = "lightblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Variability in Feature Importance Across Runs",
    subtitle = "Top 10 features by average importance",
    x = "Feature",
    y = "Importance Score"
  )


a <- multi_seed_results$selection_frequency$feature[1:20]
b <- multi_seed_results$avg_importance$feature[1:20]
keepLayers <- union(a, b)
