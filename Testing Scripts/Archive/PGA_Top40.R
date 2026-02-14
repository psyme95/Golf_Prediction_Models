#### Load Packages ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
'%not in%'  <- function(x,table) is.na(match(x,table,nomatch=NA_integer_))

#### Load Data ####
set.seed(123)
setwd("A:/OneDrive - University of Southampton/Golf")
df <- read.csv("A:/OneDrive - University of Southampton/Golf/Data/PGA_withodds.csv")
df$rating <- as.numeric(df$rating)
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
df <- df[complete.cases(df),]
eventQuality <- read.csv("A:/OneDrive - University of Southampton/Golf/Data/FieldQuality.csv")

df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID"))
  
pgalist <- split(df, df$eventID)

#### Add more stats for each tournament  #####
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

# Extract the randomly selected dataframes
proj.list <- list()
selected_indices <- sample(1:length(pgalist_filtered), 17)

for (i in selected_indices) {
  proj.list[[length(proj.list) + 1]] <- pgalist_filtered[[i]]  # Add to new list
}

# Remove data frames from original list
pgalist_filtered <- pgalist_filtered[-selected_indices]

##### Combine all tournament data into one dataframe #####
all_players_data <- data.frame()

for (eventID in names(pgalist_filtered)) {
  temp_df <- pgalist_filtered[[eventID]]
  temp_df$eventID <- eventID
  all_players_data <- rbind(all_players_data, temp_df)
}

resp <- all_players_data[, "top_40"]
expl <- all_players_data[, !names(all_players_data) %in% c("eventID", "posn", "playerID","score", "win", "top_40")]

PGA_projection <- data.frame()

for (i in 1:length(proj.list)) {
  temp_df <- proj.list[[i]]
  temp_df$eventID <- proj.list[[i]]$eventID
  PGA_projection <- rbind(PGA_projection, temp_df)
}

resp.proj <- PGA_projection[, c("playerID","eventID","top_40", "Top20_Odds", "Top40_Odds")]
expl.proj <- PGA_projection[, !names(all_players_data) %in% c("eventID", "posn", "playerID","score", "win", "top_40")]

#### Variable Selection ####
names(expl)

# Models
keepLayers <- names(expl)[c(66,75,8,21,25,26,30,33,49,57,61)] # Run 1.EQ70 - Bigboss model settings - Field quality added
keepLayers <- names(expl)[c(66,75,4,10,17,25,26,30,49,61)] # Run 2.EQ70 - Bigboss model settings - Field quality added
keepLayers <- names(expl)[c(75,4,8,10,20,21,25,26,30,49,61)] # Run 3.EQ70 - Tuned model settings - Field quality added
keepLayers <- names(expl)[c(66,75,8,21,25,26,30,33,49,57,61)] # Run 4.EQ70 - Tuned model settings - Field quality added 
keepLayers <- names(expl)[c(66,75,8,21,25,26,30,33,49,57,61)] # Run 5.EQ70 - Bigboss model settings - Field quality added - Larger split than run 1
keepLayers <- names(expl)[c(66,75,8,21,25,26,30,33,49,57,61,64)] # Run 6.EQ70 - Bigboss model settings - Field quality added - Larger split than run 1
keepLayers <- names(expl)[c(66,75,8,21,25,26,30,33,49,57,61,65)] # Run 7.EQ70 - Bigboss model settings - Field quality added - Larger split than run 1
keepLayers <- c("field", "log_rating", "current_rank", "compat_rank", "Top20_Odds", "sgtee", "X_Top5", "course", "location",
                "sgt2g_top20", "Quality", "sgatg_rank", "sgtee_top20", "diff_from_min", "sgapp_top20", "Starts_HomeShare",
                "sgp_rank", "rating_median","sgapp", "Starts_Not10")  # Run 8.EQ70 - Bigboss

# drop the layers
myExpl_prediction <- expl[keepLayers]
myExpl_prediction <- sapply(myExpl_prediction, as.numeric)

#myExpl_projection <- expl.proj[keepLayers]
names(myExpl_prediction)
#names(myExpl_projection)

run  <-   "1.EQ70" # CHANGE THE RUN NUMBER
reps  <-  1 # CHANGE NO OF MODEL REPS. This should BE 5 fold for exploratory models, 1 fold for the final run.

models.to.proc = c('GAM','GLM','RF','FDA','CTA','ANN','GBM','MAXNET','XGBOOST') 

##### Variable Correlation Check ####
cor_results <- cor(myExpl_prediction,
                   use = "pairwise.complete.obs")

# View the correlation matrix
print(cor_results)
corrplot(cor_results, method = "number")

##### Convert Categorical Variables to Factors ####
# If using categorical variables you cannot use MARS or SRE as modelling options
factors.to.convert <- which(names(myExpl_prediction)=="Starts_Not10",
                            names(myExpl_prediction)=="playerID")

for(j in factors.to.convert){
  myExpl_prediction[[j]] <- as.factor(myExpl_prediction[[j]])
}

# Create Biomod Data Object
myBiomodData  <-  BIOMOD_FormatingData(
  resp.var = resp,
  expl.var = myExpl_prediction,
  #eval.resp.var = resp.test,
  #eval.expl.var = expl.test,
  resp.name = "PGA")

# Add dummy coordinates (sequential numbers)
dummy_coords <- data.frame(
  x = 1:length(myBiomodData@data.species),
  y = 1:length(myBiomodData@data.species)
)

# Update biomod data object with these coordinates
myBiomodData@coord <- dummy_coords

#### Modelling ####
# Biomod Options
user.rf <- list('_allData_allRun' = list(ntree=500, nodesize=100))
user.XGBOOST <- list('_allData_allRun' = list(nrounds = 10))
user.GAM <- list('_allData_allRun' = list( algo = 'GAM_mgcv',
                                           type = 's_smoother',
                                           k = 4,   #If GAMS not converging, change up to 4, then 5 max
                                           interaction.level = 1,    #Increase to 1 if trouble fitting models 
                                           myFormula = NULL,
                                           family = binomial(link = 'logit'),
                                           method = 'GCV.Cp',
                                           optimizer = c('outer','newton'),
                                           select = FALSE,
                                           knots = NULL,
                                           paraPen = NULL,
                                           control = list(nthreads = 1, irls.reg = 0, epsilon = 1e-07, maxit = 200, 
                                                          trace = FALSE, mgcv.tol = 1e-07, mgcv.half = 15,
                                                          rank.tol = 1.49011611938477e-08, nlm = list(ndigit=7,
                                                                                                      gradtol=1e-06, stepmax=2, steptol=1e-04, iterlim=200,
                                                                                                      check.analyticals=0), optim = list(factr=1e+07), 
                                                          newton = list(conv.tol=1e-06, maxNstep=5, maxSstep=2,
                                                                        maxHalf=30, use.svd=0),outerPIsteps = 0, 
                                                          idLinksBases = TRUE, scalePenalty = TRUE, keepData = FALSE)))
user.GLM <- list('_allData_allRun' = list(type = "quadratic",
                                          interaction.level=0))
user.SRE <- list('_allData_allRun' = list(quant=0.025))
user.MARS <- list('_allData_allRun' = list(type='quadratic',
                                           interaction.level = 0))
user.CTA <- list('_allData_allRun' = list(control = rpart.control(cp=0.005)))

user.val <- list(RF.binary.randomForest.randomForest = user.rf,
                 XGBOOST.binary.xgboost.xgboost = user.XGBOOST,
                 GAM.binary.mgcv.gam = user.GAM,
                 GLM.binary.stats.glm = user.GLM,
                 SRE.binary.biomod2.bm_SRE = user.SRE,
                 MARS.binary.earth.earth = user.MARS,
                 CTA.binary.rpart.rpart = user.CTA)

myOpt <- bm_ModelingOptions(data.type = 'binary',
                            models=c("ANN", "CTA", "FDA", "GAM", "GBM", "GLM", "MARS", "MAXNET", "RF",
                                     "SRE", "XGBOOST"),
                            strategy = "user.defined",
                            user.val = user.val,
                            bm.format = myBiomodData)


# Individual Models
myBiomodModelOut  <-  BIOMOD_Modeling(
  bm.format=myBiomodData,
  models = models.to.proc,
  OPT.strategy = "bigboss",
  #OPT.user = myOpt,
  #OPT.user.val = user.val,
  CV.nb.rep=reps, # 5 for exploratory models, but 1 or 2 for the final model
  CV.perc=0.75,   # change up to 80 if less data (as there may not be enough to fit data)
  weights=NULL,
  var.import=3,
  metric.eval = c('ROC','TSS'),
  nb.cpu = 4
  #rescal.all.models = TRUE
)


##### Evaluate the Single Models #####
myBiomodModelOut 

myBiomodModelEval  <-  get_evaluations(myBiomodModelOut)
myBiomodModelEval

dplyr::filter(myBiomodModelEval, algo=="RF" & metric.eval=="TSS")

varimp <- get_variables_importance(myBiomodModelOut)
print(varimp)

aggregateVarImp <- stats::aggregate(varimp[, "var.imp"], by=list( varimp$run, varimp$algo, varimp$expl.var), mean)
colnames(aggregateVarImp) <- c("Run","Algorithm","ExplanatoryVar","Mean")

aggregateVarImp <- aggregateVarImp[order(aggregateVarImp$Run,aggregateVarImp$Algorithm),]
write.csv(aggregateVarImp,paste0("A:/OneDrive - University of Southampton/Golf/Evaluation/Variable_Importance_Run", run,".csv"))

array_of_model_results_new <- as.data.frame(myBiomodModelEval[which(myBiomodModelEval$run=="allRun"),])
write.csv(array_of_model_results_new,file=paste0("A:/Onedrive - University of Southampton/Golf/Evaluation/Individual_Model_Evaluations_Run", run, ".csv")) 

#create TSS scores table
allRun.model.names <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"full.name"]
allRun.model.sens <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"sensitivity"]
allRun.model.spec <- myBiomodModelEval[which(myBiomodModelEval$run=="allRun"&myBiomodModelEval$metric.eval=="TSS"),"specificity"]
allRun.model.TSS <- (allRun.model.sens/100+allRun.model.spec/100)-1

df.metrics  <-  data.frame(matrix(0, ncol = length(allRun.model.TSS), nrow = 1))
names(df.metrics) <- allRun.model.names
row.names(df.metrics) <- "TSS"
df.metrics[1,1:length(allRun.model.TSS)] <- allRun.model.TSS

#  Plot the response plots
proj_dir <- paste0(getwd(),"/BIOMOD2_plots/Run_",run,"/")# make sure there is a BIOMOD2_plots folder in Area folder
if (!dir.exists(proj_dir)) {
  dir.create(proj_dir)
}

for(i in myBiomodModelOut@models.computed[grep("allRun",myBiomodModelOut@models.computed)]){
  png(paste0("A:/OneDrive - University of Southampton/Golf/Biomod2_plots/Run_",run,"/", i, "_response_plot.png"),width=1024,height=1024)
  bm_PlotResponseCurves(myBiomodModelOut, models.chosen = i,do.bivariate=F,fixed.var="mean")
  dev.off()
}

#### Create the ENSEMBLE MODEL ####
gc()

myBiomodEM  <-  biomod2::BIOMOD_EnsembleModeling(
  bm.mod = myBiomodModelOut,
  models.chosen = allRun.model.names, #Include all FULL MODELS
  em.algo = c('EMwmean'), #'EMci','EMcv'), #('EMmean', 'EMcv', 'EMci', 'EMmedian', 'EMca' or 'EMwmean') ('prob.median', 'prob.cv', 'prob.ci', 'committee.averaging', 'prob.mean.weight')
  metric.select = c("user.defined"),
  metric.select.thresh = c(0.2), #lowered to 0.5 because 0.6 is difficult to achieve.  If ensemmble is poor though, change to 0.6
  metric.select.table = df.metrics,
  metric.eval = c('TSS', 'ROC'),
  var.import = 3,
  EMwmean.decay = 'proportional',
)

myBiomodEMEval  <-  get_evaluations(myBiomodEM)


#  Plot the response plots
proj_dir <- paste0(getwd(),"/BIOMOD2_plots/Ensemble_Run_",run,"/")# make sure there is a BIOMOD2_plots folder in Area folder
if (!dir.exists(proj_dir)) {
  dir.create(proj_dir)
}

png(paste0("A:/OneDrive - University of Southampton/Golf/Biomod2_plots/Ensemble_Run_",run,"/Ensemble_response_plot.png"),width=1024,height=1024)
bm_PlotResponseCurves(myBiomodEM, models.chosen = "all")
dev.off()


##### Evaluate the ensemble model performance #####
myBiomodEM
ensemble.Model.Eval <- get_evaluations(myBiomodEM)
print(ensemble.Model.Eval)

write.csv(ensemble.Model.Eval, file=paste0("A:/OneDrive - University of Southampton/Golf/Evaluation/Ensemble_Model_Evaluations_Run", run, ".csv"))

##### Find threshold values for target sensitivity/specificity
source("A:/OneDrive - University of Southampton/Golf/R Scripts/cutoff_finder_biomod.R")

result <- cutoff_finder_biomod(
  biomod_ensemble = myBiomodEM,
  biomod_data_object = myBiomodData,
  sensitivity_target = 0.95,
  specificity_target = 0.95
)

# View full performance metrics
sensspecthreshold <- print(result$full_performance)
write.csv(sensspecthreshold, paste0("A:/OneDrive - University of Southampton/Golf/Evaluation/Run", run, "_Threshold_Evaluation_Scores.csv"), row.names=F)

# View threshold closest to 95% sensitivity
print(result$target_sensitivity_threshold)

# View threshold closest to 95% specificity
print(result$target_specificity_threshold)

# Plotting both sensitivity and specificity
plot(result$full_performance$threshold, result$full_performance$sensitivity, 
     type = 'l', col = 'blue', 
     xlab = 'Threshold', ylab = 'Percentage',
     ylim = c(0, 100))
lines(result$full_performance$threshold, result$full_performance$specificity, 
      col = 'red')
abline(h = 95, lty = 2, col = 'green')
legend('topright', 
       legend = c('Sensitivity', 'Specificity', '95% Line'), 
       col = c('blue', 'red', 'green'), 
       lty = c(1, 1, 2))

# ####  Project the ensemble model to testing tournament as check how good it is ####
resp.proj
resp.proj.xy <- resp.proj[,1:2]
expl.proj

myExpl_projection <- expl.proj[keepLayers]
names(myExpl_projection)

# Ensemble forecasting remains the same
myBiomodProj <- BIOMOD_EnsembleForecasting(
  bm.em = myBiomodEM,
  bm.proj = NULL,
  proj.name = "wmean",
  new.env = myExpl_projection,
  new.env.xy = resp.proj.xy,
  models.chosen = "all",
  metric.binary = "TSS",
  metric.filter = "TSS",
  na.rm = TRUE
)

# Format prediction data
PGA_Prediction <- cbind(myBiomodProj@coord, myBiomodProj@proj.out@val$pred)
colnames(PGA_Prediction)[3] <- "Model_Score"
PGA_Pred_list <- split(PGA_Prediction, PGA_Prediction$eventID)

# Define threshold ranges 
threshold_ranges <- list(
  list(start = 1, end = 10),
  list(start = 1, end = 20),
  list(start = 1, end = 30),
  list(start = 1, end = 40),
  list(start = 10, end = 50),
  list(start = 10, end = 20),
  list(start = 10, end = 30),
  list(start = 10, end = 40),
  list(start = 20, end = 60),
  list(start = 20, end = 40),
  list(start = 20, end = 50),
  list(start = 20, end = 30),
  list(start = 30, end = 40),
  list(start = 30, end = 50),
  list(start = 30, end = 60),
  list(start = 40, end = 50),
  list(start = 40, end = 60)
)

fixed_stake_amount <- 10
target_profit <- 10
stake_types <- c("fixed", "flex")

#### Process each event ####
for (i in 1:length(PGA_Pred_list)) {
  # Print event ID for debugging
  cat("Processing event", i, "of", length(PGA_Pred_list), "\n")
  
  # Make sure PGA_Pred_list[[i]] has data
  if (nrow(PGA_Pred_list[[i]]) == 0) {
    cat("Warning: Empty data frame for event", i, "\n")
    next
  }
  
  # Join prediction data with actual results - use dplyr for clearer debugging
  event_data <- PGA_Pred_list[[i]]
  projection_data <- PGA_projection[PGA_projection$eventID %in% unique(event_data$eventID), 
                                    c("eventID", "playerID", "Top40_Odds", "posn")]
  
  # Debug print
  cat("Event data rows:", nrow(event_data), "Projection data rows:", nrow(projection_data), "\n")
  
  # Check that projection data has rows
  if (nrow(projection_data) == 0) {
    cat("Warning: No matching projection data for event", i, "\n")
    next
  }
  
  # Perform the merge with additional error handling
  merged_data <- merge(
    event_data,
    projection_data,
    by = c("eventID", "playerID"),
    all.x = TRUE
  )
  
  # Check merge result
  if (nrow(merged_data) == 0) {
    cat("Warning: Merge resulted in 0 rows for event", i, "\n")
    next
  }
  
  # Update PGA_Pred_list
  PGA_Pred_list[[i]] <- merged_data
  
  # Add actual Top40 finish column - ensure posn is not NA
  PGA_Pred_list[[i]]$Top40 <- ifelse(!is.na(PGA_Pred_list[[i]]$posn) & PGA_Pred_list[[i]]$posn <= 40, 1, 0)
  
  # Setup odds validity - ensure Top40_Odds is not NA
  PGA_Pred_list[[i]]$Valid_Odds <- !is.na(PGA_Pred_list[[i]]$Top40_Odds) & PGA_Pred_list[[i]]$Top40_Odds > 1
  PGA_Pred_list[[i]]$Fixed_Stake <- fixed_stake_amount
  PGA_Pred_list[[i]]$Flex_Stake <- ifelse(
    PGA_Pred_list[[i]]$Valid_Odds,
    round(target_profit / (PGA_Pred_list[[i]]$Top40_Odds - 1), 2),
    0
  )
  
  # Calculate thresholds and create model score columns for each range
  for (range_idx in 1:length(threshold_ranges)) {
    range <- threshold_ranges[[range_idx]]
    start_val <- range$start
    end_val <- range$end
    
    # Create a range label for columns
    range_label <- paste0(start_val, "to", end_val)
    
    # Get the total number of players in the event
    total_players <- nrow(PGA_Pred_list[[i]])
    
    # Safely determine start and end threshold values based on ranks
    sorted_scores <- sort(PGA_Pred_list[[i]]$Model_Score, decreasing = TRUE)
    
    # Handle case where there aren't enough players
    if (length(sorted_scores) < start_val) {
      cat("Warning: Not enough players for start threshold in event", i, "\n")
      start_threshold_value <- min(sorted_scores, na.rm = TRUE)
    } else {
      start_threshold_value <- sorted_scores[min(start_val, total_players)]
    }
    
    if (length(sorted_scores) < end_val) {
      cat("Warning: Not enough players for end threshold in event", i, "\n")
      end_threshold_value <- min(sorted_scores, na.rm = TRUE)
    } else {
      end_threshold_value <- sorted_scores[min(end_val, total_players)]
    }
    
    # Create model score column - Note we need to reverse the comparison
    # since higher ranked players (lower index) have higher scores
    col_name <- paste0("Range", range_label, "_ModelScore")
    PGA_Pred_list[[i]][[col_name]] <- ifelse(
      PGA_Pred_list[[i]]$Model_Score >= end_threshold_value & 
        PGA_Pred_list[[i]]$Model_Score <= start_threshold_value, 1, 0)
    
    # Calculate if model was correct
    correct_col <- paste0("Range", range_label, "_ModelScore_Correct")
    PGA_Pred_list[[i]][[correct_col]] <- ifelse(PGA_Pred_list[[i]]$Top40 == PGA_Pred_list[[i]][[col_name]], 1, 0)
    
    # Process each stake type
    for (stake_type in stake_types) {
      stake_col_suffix <- ifelse(stake_type == "fixed", ", ", "_Flex")
      stake_amount_col <- ifelse(stake_type == "fixed", "Fixed_Stake", "Flex_Stake")
      
      # Calculate stake placed
      stake_placed_col <- paste0("Range", range_label, "_Stake_Placed", stake_col_suffix)
      PGA_Pred_list[[i]][[stake_placed_col]] <- ifelse(
        PGA_Pred_list[[i]][[col_name]] == 1 & PGA_Pred_list[[i]]$Valid_Odds,
        PGA_Pred_list[[i]][[stake_amount_col]],
        0
      )
      
      # Calculate winnings
      winnings_col <- paste0("Range", range_label, "_Player_Winnings", stake_col_suffix)
      PGA_Pred_list[[i]][[winnings_col]] <- ifelse(
        PGA_Pred_list[[i]][[col_name]] == 1 & 
          PGA_Pred_list[[i]]$Valid_Odds & 
          PGA_Pred_list[[i]][[correct_col]] == 1,
        PGA_Pred_list[[i]][[stake_amount_col]] * PGA_Pred_list[[i]]$Top40_Odds,
        0
      )
      
      # Calculate profit
      profit_col <- paste0("Range", range_label, "_Player_Profit", stake_col_suffix)
      PGA_Pred_list[[i]][[profit_col]] <- PGA_Pred_list[[i]][[winnings_col]] - PGA_Pred_list[[i]][[stake_placed_col]]
    }
  }
}

# Create summaries
all_summaries <- data.frame()

# Function to create column name for summary metrics
col_name <- function(metric, range_label, stake_type = NULL) {
  if (is.null(stake_type)) {
    return(paste0(metric, "_range", range_label))
  } else {
    return(paste0(metric, "_", stake_type, "_range", range_label))
  }
}

# Process each event for summary statistics
for (i in 1:length(PGA_Pred_list)) {
  # Skip if the event has no data
  if (is.null(PGA_Pred_list[[i]]) || nrow(PGA_Pred_list[[i]]) == 0) {
    cat("Skipping summary for event", i, "- no data\n")
    next
  }
  
  event_id <- unique(PGA_Pred_list[[i]]$eventID)[1]
  valid_data <- PGA_Pred_list[[i]][PGA_Pred_list[[i]]$Valid_Odds, ]
  
  # Skip if no valid odds
  if (nrow(valid_data) == 0) {
    cat("Skipping summary for event", i, "- no valid odds\n")
    next
  }
  
  # Initialize event summary with event ID
  event_summary <- data.frame(eventID = event_id)
  
  # Calculate summaries for each threshold range
  for (range_idx in 1:length(threshold_ranges)) {
    range <- threshold_ranges[[range_idx]]
    range_label <- paste0(range$start, "to", range$end)
    
    # Base metrics
    model_score_col <- paste0("Range", range_label, "_ModelScore")
    correct_col <- paste0("Range", range_label, "_ModelScore_Correct")
    
    # Check if columns exist
    if (!(model_score_col %in% names(valid_data)) || !(correct_col %in% names(valid_data))) {
      cat("Warning: Missing columns for range", range_label, "in event", i, "\n")
      next
    }
    
    # Calculate betting totals and success rates
    total_bets <- sum(valid_data[[model_score_col]] == 1, na.rm = TRUE)
    successful_bets <- sum(valid_data[[model_score_col]] == 1 & valid_data[[correct_col]] == 1, na.rm = TRUE)
    success_rate <- ifelse(total_bets > 0, round(successful_bets / total_bets * 100, 2), 0)
    
    # Add to summary
    event_summary[[col_name("bets", range_label)]] <- total_bets
    event_summary[[col_name("successful", range_label)]] <- successful_bets
    event_summary[[col_name("success_rate", range_label)]] <- success_rate
    
    # Calculate for each stake type
    for (stake_type in stake_types) {
      stake_suffix <- ifelse(stake_type == "fixed", ", ", "_Flex")
      
      # Get column names
      stake_col <- paste0("Range", range_label, "_Stake_Placed", stake_suffix)
      winnings_col <- paste0("Range", range_label, "_Player_Winnings", stake_suffix)
      profit_col <- paste0("Range", range_label, "_Player_Profit", stake_suffix)
      
      # Check if columns exist
      if (!(stake_col %in% names(valid_data)) || 
          !(winnings_col %in% names(valid_data)) || 
          !(profit_col %in% names(valid_data))) {
        cat("Warning: Missing columns for stake type", stake_type, "in range", range_label, "event", i, "\n")
        next
      }
      
      # Calculate totals
      total_stake <- sum(valid_data[[stake_col]], na.rm = TRUE)
      total_winnings <- sum(valid_data[[winnings_col]], na.rm = TRUE)
      total_profit <- sum(valid_data[[profit_col]], na.rm = TRUE)
      roi <- ifelse(total_stake > 0, round(total_profit / total_stake * 100, 2), 0)
      
      # Add to summary
      event_summary[[col_name("stake", range_label, stake_type)]] <- round(total_stake, 2)
      event_summary[[col_name("winnings", range_label, stake_type)]] <- round(total_winnings, 2)
      event_summary[[col_name("profit", range_label, stake_type)]] <- round(total_profit, 2)
      event_summary[[col_name("roi", range_label, stake_type)]] <- roi
    }
  }
  
  # Add the event summary to the combined dataframe
  all_summaries <- rbind(all_summaries, event_summary)
  
  # Store summary as attribute
  attr(PGA_Pred_list[[i]], "summary") <- event_summary
}

# Create separate workbooks for summaries and bets
summary_wb <- createWorkbook()
bets_wb <- createWorkbook()

# Add summary sheet to summary workbook
addWorksheet(summary_wb, "All_Event_Summaries")
writeData(summary_wb, "All_Event_Summaries", all_summaries)

# Function to calculate confidence interval
calculate_ci <- function(values, confidence = 0.95) {
  if (length(values) < 2 || all(is.na(values))) {
    return(list(lower = NA, upper = NA))
  }
  
  # Remove NA values
  values <- values[!is.na(values)]
  
  if (length(values) < 2) {
    return(list(lower = NA, upper = NA))
  }
  
  # Calculate standard error
  se <- sd(values) / sqrt(length(values))
  
  # t value for confidence interval
  t_value <- qt((1 + confidence) / 2, df = length(values) - 1)
  
  # Calculate confidence interval
  mean_val <- mean(values)
  margin <- t_value * se
  
  return(list(lower = mean_val - margin, upper = mean_val + margin))
}

# Create a consolidated range summary showing average success_rate and ROI for each range
range_summary <- data.frame(
  Range = character(),
  Success_Rate = numeric(),
  ROI_Fixed = numeric(),
  ROI_Flex = numeric(),
  ROI_Fixed_SD = numeric(),
  ROI_Flex_SD = numeric(),
  ROI_Fixed_CI_Lower = numeric(),
  ROI_Fixed_CI_Upper = numeric(), 
  ROI_Flex_CI_Lower = numeric(),
  ROI_Flex_CI_Upper = numeric(),
  Total_Bets = numeric(),
  Total_Events = numeric(),
  stringsAsFactors = FALSE
)

for (range_idx in 1:length(threshold_ranges)) {
  range <- threshold_ranges[[range_idx]]
  range_label <- paste0(range$start, "to", range$end)
  
  # Get column names for metrics we want to average
  success_rate_col <- col_name("success_rate", range_label)
  roi_fixed_col <- col_name("roi", range_label, "fixed")
  roi_flex_col <- col_name("roi", range_label, "flex")
  bets_col <- col_name("bets", range_label)
  
  # Check if we have data for this range
  if (!(success_rate_col %in% names(all_summaries)) || 
      !(roi_fixed_col %in% names(all_summaries)) ||
      !(roi_flex_col %in% names(all_summaries)) ||
      !(bets_col %in% names(all_summaries))) {
    cat("Skipping range summary for", range_label, "- no data\n")
    next
  }
  
  # Calculate totals and averages
  total_bets <- sum(all_summaries[[bets_col]], na.rm = TRUE)
  
  # Create properly weighted averages - weight success rate and ROI by number of bets
  success_rate_avg <- 0
  roi_fixed_avg <- 0
  roi_flex_avg <- 0
  
  # Initialize standard deviation and confidence interval values
  roi_fixed_sd <- 0
  roi_flex_sd <- 0
  roi_fixed_ci_lower <- NA
  roi_fixed_ci_upper <- NA
  roi_flex_ci_lower <- NA
  roi_flex_ci_upper <- NA
  
  if (total_bets > 0) {
    # Weight by number of bets for more accurate average
    success_rate_avg <- sum(all_summaries[[success_rate_col]] * all_summaries[[bets_col]], na.rm = TRUE) / total_bets
    
    # For ROI, we need to calculate total profit and total stake across all events
    stake_fixed_col <- col_name("stake", range_label, "fixed")
    profit_fixed_col <- col_name("profit", range_label, "fixed")
    total_stake_fixed <- sum(all_summaries[[stake_fixed_col]], na.rm = TRUE)
    total_profit_fixed <- sum(all_summaries[[profit_fixed_col]], na.rm = TRUE)
    
    stake_flex_col <- col_name("stake", range_label, "flex")
    profit_flex_col <- col_name("profit", range_label, "flex")
    total_stake_flex <- sum(all_summaries[[stake_flex_col]], na.rm = TRUE)
    total_profit_flex <- sum(all_summaries[[profit_flex_col]], na.rm = TRUE)
    
    roi_fixed_avg <- ifelse(total_stake_fixed > 0, (total_profit_fixed / total_stake_fixed) * 100, 0)
    roi_flex_avg <- ifelse(total_stake_flex > 0, (total_profit_flex / total_stake_flex) * 100, 0)
    
    # Calculate ROI for each individual event (to compute standard deviation and confidence intervals)
    event_roi_fixed <- sapply(1:nrow(all_summaries), function(j) {
      stake <- all_summaries[j, stake_fixed_col]
      profit <- all_summaries[j, profit_fixed_col]
      ifelse(stake > 0 && !is.na(stake) && !is.na(profit), (profit / stake) * 100, NA)
    })
    
    event_roi_flex <- sapply(1:nrow(all_summaries), function(j) {
      stake <- all_summaries[j, stake_flex_col]
      profit <- all_summaries[j, profit_flex_col]
      ifelse(stake > 0 && !is.na(stake) && !is.na(profit), (profit / stake) * 100, NA)
    })
    
    # Only include events that had bets for this range
    valid_fixed_roi <- event_roi_fixed[!is.na(event_roi_fixed) & all_summaries[[bets_col]] > 0]
    valid_flex_roi <- event_roi_flex[!is.na(event_roi_flex) & all_summaries[[bets_col]] > 0]
    
    # Calculate standard deviation
    roi_fixed_sd <- ifelse(length(valid_fixed_roi) > 1, sd(valid_fixed_roi, na.rm = TRUE), 0)
    roi_flex_sd <- ifelse(length(valid_flex_roi) > 1, sd(valid_flex_roi, na.rm = TRUE), 0)
    
    # Calculate confidence intervals
    if (length(valid_fixed_roi) > 1) {
      fixed_ci <- calculate_ci(valid_fixed_roi, 0.95)
      roi_fixed_ci_lower <- fixed_ci$lower
      roi_fixed_ci_upper <- fixed_ci$upper
    }
    
    if (length(valid_flex_roi) > 1) {
      flex_ci <- calculate_ci(valid_flex_roi, 0.95)
      roi_flex_ci_lower <- flex_ci$lower
      roi_flex_ci_upper <- flex_ci$upper
    }
  }
  
  # Count events with this range
  events_with_range <- sum(all_summaries[[bets_col]] > 0, na.rm = TRUE)
  
  # Add to range summary
  range_summary <- rbind(range_summary, data.frame(
    Range = paste(range$start, "-", range$end),
    Success_Rate = round(success_rate_avg, 2),
    ROI_Fixed = round(roi_fixed_avg, 2),
    ROI_Flex = round(roi_flex_avg, 2),
    ROI_Fixed_SD = round(roi_fixed_sd, 2),
    ROI_Flex_SD = round(roi_flex_sd, 2),
    ROI_Fixed_CI_Lower = round(roi_fixed_ci_lower, 2),
    ROI_Fixed_CI_Upper = round(roi_fixed_ci_upper, 2),
    ROI_Flex_CI_Lower = round(roi_flex_ci_lower, 2),
    ROI_Flex_CI_Upper = round(roi_flex_ci_upper, 2),
    Total_Bets = total_bets,
    Total_Events = events_with_range,
    stringsAsFactors = FALSE
  ))
}

# Sort the range summary by ROI_Fixed (descending)
range_summary <- range_summary[order(-range_summary$ROI_Fixed), ]

# Add the range summary overview sheet
addWorksheet(summary_wb, "Range_Performance_Summary")
writeData(summary_wb, "Range_Performance_Summary", range_summary)

# Create separate sheets for each range in the summary workbook
for (range_idx in 1:length(threshold_ranges)) {
  range <- threshold_ranges[[range_idx]]
  range_label <- paste0(range$start, "to", range$end)
  
  # Create sheet name
  sheet_name <- paste0("Range", range_label, "_Summary")
  
  # Filter columns relevant to this range
  prefix <- paste0("_range", range_label)
  fixed_prefix <- paste0("_fixed_range", range_label)
  flex_prefix <- paste0("_flex_range", range_label)
  
  # Check if any columns match our patterns
  matching_cols <- c(
    "eventID",
    names(all_summaries)[grepl(prefix, names(all_summaries)) & !grepl("fixed|flex", names(all_summaries))],
    names(all_summaries)[grepl(fixed_prefix, names(all_summaries))],
    names(all_summaries)[grepl(flex_prefix, names(all_summaries))]
  )
  
  # Skip if we don't have any matching columns
  if (length(matching_cols) <= 1) {  # Only eventID would be 1
    cat("Skipping sheet", sheet_name, "- no matching columns\n")
    next
  }
  
  # Create summary dataframe for this range
  range_summary <- all_summaries[, matching_cols]
  
  # Add worksheet and write data
  addWorksheet(summary_wb, sheet_name)
  writeData(summary_wb, sheet_name, range_summary)
}

# Extract the key columns we want to include for every sheet
base_columns <- c("eventID", "playerID", "Model_Score", "Top40_Odds", "posn", "Top40")

# Create consolidated bet data for each range in the bets workbook
player_bets_by_range <- list()
for (range_idx in 1:length(threshold_ranges)) {
  range <- threshold_ranges[[range_idx]]
  range_label <- paste0(range$start, "to", range$end)
  
  # Create sheet name
  sheet_name <- paste0("Range", range_label, "_Bets")
  
  # Collect columns specific to this range
  range_columns <- c(
    paste0("Range", range_label, "_ModelScore"),
    paste0("Range", range_label, "_ModelScore_Correct"),
    paste0("Range", range_label, "_Stake_Placed"),
    paste0("Range", range_label, "_Player_Winnings"),
    paste0("Range", range_label, "_Player_Profit"),
    paste0("Range", range_label, "_Stake_Placed_Flex"),
    paste0("Range", range_label, "_Player_Winnings_Flex"),
    paste0("Range", range_label, "_Player_Profit_Flex")
  )
  
  # Columns to extract
  columns_to_extract <- c(base_columns, range_columns)
  
  # Create empty dataframe for this range
  player_bets_by_range[[range_idx]] <- data.frame()
  
  # Extract data from each event and filter to only include actual bets
  for (i in 1:length(PGA_Pred_list)) {
    # Skip if no data for this event
    if (is.null(PGA_Pred_list[[i]]) || nrow(PGA_Pred_list[[i]]) == 0) {
      next
    }
    
    # Get data for this event
    event_data <- PGA_Pred_list[[i]]
    
    # Check if the columns exist
    model_score_col <- paste0("Range", range_label, "_ModelScore")
    if (!(model_score_col %in% names(event_data))) {
      next
    }
    
    # Filter to only include players with Valid_Odds and where a ModelScore = 1 (actual bets placed)
    try({
      bet_data <- event_data[event_data$Valid_Odds & event_data[[model_score_col]] == 1, 
                             intersect(columns_to_extract, names(event_data))]
      
      # Add to combined dataframe if there are any bets
      if (nrow(bet_data) > 0) {
        if (nrow(player_bets_by_range[[range_idx]]) == 0) {
          player_bets_by_range[[range_idx]] <- bet_data
        } else {
          # Make sure columns match before binding
          cols_to_use <- intersect(names(player_bets_by_range[[range_idx]]), names(bet_data))
          player_bets_by_range[[range_idx]] <- rbind(
            player_bets_by_range[[range_idx]][, cols_to_use], 
            bet_data[, cols_to_use]
          )
        }
      }
    }, silent = TRUE)
  }
  
  # Add worksheet and write data if we have any bets for this range
  if (!is.null(player_bets_by_range[[range_idx]]) && nrow(player_bets_by_range[[range_idx]]) > 0) {
    addWorksheet(bets_wb, sheet_name)
    writeData(bets_wb, sheet_name, player_bets_by_range[[range_idx]])
  }
}

# Add sheet with all players and predictions (not just bets) to bets workbook
# Filter out any NULL or empty data frames before combining
valid_pred_list <- PGA_Pred_list[sapply(PGA_Pred_list, function(x) !is.null(x) && nrow(x) > 0)]
if (length(valid_pred_list) > 0) {
  # Find common columns across all data frames
  common_cols <- Reduce(intersect, lapply(valid_pred_list, names))
  
  # Only combine if we have common columns
  if (length(common_cols) > 0) {
    all_players_data <- do.call(rbind, lapply(valid_pred_list, function(x) x[, common_cols]))
    addWorksheet(bets_wb, "All_Players_Data")
    writeData(bets_wb, "All_Players_Data", all_players_data)
  }
}

# Save the workbooks
summary_excel_file_path <- paste0("A:/OneDrive - University of Southampton/Golf/Betting Simulations/PGA_Event_Betting_Outcomes_Model", run, ".xlsx")
bets_excel_file_path <- paste0("A:/OneDrive - University of Southampton/Golf/Betting Simulations/PGA_Player_Betting_Outcomes_Model", run, ".xlsx")

saveWorkbook(summary_wb, summary_excel_file_path, overwrite = TRUE)
saveWorkbook(bets_wb, bets_excel_file_path, overwrite = TRUE)