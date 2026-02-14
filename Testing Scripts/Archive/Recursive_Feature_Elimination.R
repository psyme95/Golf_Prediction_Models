# PGA Golf Data - Recursive Feature Elimination (RFE) Analysis
# This script performs comprehensive RFE analysis on PGA golf data

# Load required libraries
library(caret)
library(randomForest)
library(e1071)
library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(VIM)
library(doParallel)
library(glmnet)

# Set up parallel processing for faster computation
cl <- makeCluster(8)
registerDoParallel(cl)

# Load the data
cat("Loading PGA data...\n")
pga_data <- read_excel("./Data/PGA_processed_features.xlsx")
metadata <- read.csv("./Data/PGA_processed_features_metadata.csv", stringsAsFactors = FALSE)
pga_data <- pga_data[complete.cases(pga_data),]

# Define target variable (assuming 'win' is the target based on metadata)
target_var <- "top_20"
if(!target_var %in% names(pga_data)) {
  # If 'win' doesn't exist, look for other potential targets
  potential_targets <- c("posn", "score", "top5", "top10", "top20")
  target_var <- potential_targets[potential_targets %in% names(pga_data)][1]
  cat("Using", target_var, "as target variable\n")
}

# Prepare features and target
feature_cols <- setdiff(names(pga_data), c(target_var, "eventID", "playerID", "top_40", "top_10", "top_5", "win", "Top20_Profit", "Top40_Profit", "Date", "posn", "score", "Event.Name"))
X <- pga_data[, feature_cols]
y <- pga_data[[target_var]]

# Convert target to factor if it's binary/categorical
if(target_var == "win" || length(unique(y)) <= 10) {
  y <- as.factor(y)
  problem_type <- "classification"
} else {
  problem_type <- "regression"
}

cat("Problem type:", problem_type, "\n")
cat("Number of features for RFE:", ncol(X), "\n")

# Remove near-zero variance predictors
cat("\nRemoving near-zero variance predictors...\n")
nzv <- nearZeroVar(X, saveMetrics = TRUE)
if(sum(nzv$nzv) > 0) {
  cat("Removing", sum(nzv$nzv), "near-zero variance predictors\n")
  X <- X[, !nzv$nzv]
}

# Handle highly correlated features (optional - comment out if you want to keep all)
cat("\nChecking for highly correlated features...\n")
numeric_cols <- sapply(X, is.numeric)
if(sum(numeric_cols) > 1) {
  cor_matrix <- cor(X[, numeric_cols], use = "complete.obs")
  high_cor <- findCorrelation(cor_matrix, cutoff = 0.75, verbose = TRUE)
  if(length(high_cor) > 0) {
    cat("Removing", length(high_cor), "highly correlated features\n")
    X <- X[, -high_cor]
  }
}

cat("Final number of features:", ncol(X), "\n")

# Split data into training and testing sets
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE, times = 1)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Define control parameters for RFE
rfe_ctrl <- rfeControl(
  functions = if(problem_type == "classification") rfFuncs else rfFuncs,
  method = "cv",
  number = 5,  # 5-fold cross-validation
  verbose = TRUE,
  allowParallel = TRUE
)

# Define the range of features to test
feature_sizes <- c(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 75, 100)
feature_sizes <- feature_sizes[feature_sizes <= ncol(X_train)]

cat("\nStarting Recursive Feature Elimination...\n")
cat("Testing feature sizes:", paste(feature_sizes, collapse = ", "), "\n")

# Perform RFE with Random Forest
set.seed(123)
start_time <- Sys.time()
rfe_results <- rfe(
  x = X_train,
  y = y_train,
  sizes = feature_sizes,
  rfeControl = rfe_ctrl
)
end_time <- Sys.time()

cat("RFE completed in", round(difftime(end_time, start_time, units = "mins"), 2), "minutes\n")

# Print RFE results
print(rfe_results)

# Plot RFE results
plot(rfe_results, type = c("g", "o"), main = "RFE Results: Model Performance vs Number of Features")

# Get optimal number of features
optimal_features <- rfe_results$optsize
cat("\nOptimal number of features:", optimal_features, "\n")

# Get the selected features
selected_features <- rfe_results$optVariables
cat("Selected features:\n")
for(i in 1:length(selected_features)) {
  feature_name <- selected_features[i]
  # Get description from metadata if available
  description <- metadata$Explanation[metadata$Variables == feature_name]
  if(length(description) > 0 && !is.na(description)) {
    cat(sprintf("%2d. %s: %s\n", i, feature_name, description))
  } else {
    cat(sprintf("%2d. %s\n", i, feature_name))
  }
}

# Create final model with selected features
cat("\nTraining final model with selected features...\n")
X_train_selected <- X_train[, selected_features]
X_test_selected <- X_test[, selected_features]

if(problem_type == "classification") {
  final_model <- randomForest(X_train_selected, y_train, importance = TRUE)
  pred_test <- predict(final_model, X_test_selected)
  
  # Model performance
  conf_matrix <- confusionMatrix(pred_test, y_test)
  print(conf_matrix)
  
  cat("\nAccuracy:", round(conf_matrix$overall['Accuracy'], 4), "\n")
  
} else {
  final_model <- randomForest(X_train_selected, y_train, importance = TRUE)
  pred_test <- predict(final_model, X_test_selected)
  
  # Model performance
  rmse <- sqrt(mean((pred_test - y_test)^2))
  mae <- mean(abs(pred_test - y_test))
  r2 <- cor(pred_test, y_test)^2
  
  cat("\nRegression Performance:\n")
  cat("RMSE:", round(rmse, 4), "\n")
  cat("MAE:", round(mae, 4), "\n")
  cat("R-squared:", round(r2, 4), "\n")
}

# Feature importance plot
importance_df <- data.frame(
  Feature = rownames(importance(final_model)),
  Importance = importance(final_model)[, 1]
)
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

ggplot(importance_df[1:min(50, nrow(importance_df)), ], 
       aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 20 Feature Importance (Final Model)",
       x = "Features", y = "Importance") +
  theme_minimal()

# Create a summary table of selected features with their categories
if(nrow(metadata) > 0) {
  selected_summary <- data.frame(
    Feature = selected_features,
    stringsAsFactors = FALSE
  )
  
  selected_summary <- merge(selected_summary, metadata, by.x = "Feature", by.y = "Variables", all.x = TRUE)
  selected_summary <- selected_summary[match(selected_features, selected_summary$Feature), ]
  
  cat("\nSelected Features Summary by Category:\n")
  category_summary <- table(selected_summary$Category)
  print(category_summary)
  
  # Save results
  write.csv(selected_summary, "selected_features_summary.csv", row.names = FALSE)
  cat("\nSelected features summary saved to 'selected_features_summary.csv'\n")
}