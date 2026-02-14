#### Load Packages ####
library(dplyr)
library(biomod2)
library(data.table)
library(rpart)
library(corrplot)
library(openxlsx)
'%not in%'  <- function(x,table) is.na(match(x,table,nomatch=NA_integer_))

#### Load Data ####
setwd("A:/OneDrive - University of Southampton/Golf")

df <- read.csv("./Data/PGA_withodds.csv")
df$rating <- as.numeric(df$rating)
df[is.na(df$Top20_Odds), "Top20_Odds"] <- 0
df[is.na(df$Top40_Odds), "Top40_Odds"] <- 0
df <- df[complete.cases(df),]
eventQuality <- read.csv("./Data/FieldQuality.csv")

df <- df %>%
  left_join(eventQuality, by = c("eventID" = "EventID")) %>%
  filter(Quality<= 71)
  
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
  pgalist[[eventID]]$top_20 <- ifelse(pgalist[[eventID]]$posn <= 20, 1, 0)
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

resp <- all_players_data[, "top_20"]
expl <- all_players_data[, !names(all_players_data) %in% c("eventID", "posn", "playerID","score", "win", "top_20")]

PGA_projection <- data.frame()

for (i in 1:length(proj.list)) {
  temp_df <- proj.list[[i]]
  temp_df$eventID <- proj.list[[i]]$eventID
  PGA_projection <- rbind(PGA_projection, temp_df)
}

resp.proj <- PGA_projection[, c("playerID","eventID","top_20", "Top20_Odds", "Top40_Odds")]
expl.proj <- PGA_projection[, !names(all_players_data) %in% c("eventID", "posn", "playerID","score", "win", "top_20")]

#### Variable Selection ####
names(expl)

# Models
keepLayers <- names(expl)[c(66,75,8,21,25,26,30,33,49,57,61)] # Run 1.EQ70.Top20 - Bigboss model settings - Field quality added - Larger split than run 1
keepLayers <- names(expl)[c(66,75,7,8,21,25,26,30,33,49,61,64)] # Run 2.EQ71.Top20 - Bigboss model settings - Field quality added - Larger split than run 1

# drop the layers
myExpl_prediction <- expl[keepLayers]
#myExpl_projection <- expl.proj[keepLayers]
names(myExpl_prediction)
#names(myExpl_projection)

run  <-   "2.EQ_Over71.Top20" # CHANGE THE RUN NUMBER
reps  <-  1 # CHANGE NO OF MODEL REPS. This should BE 5 fold for exploratory models, 1 fold for the final run.

models.to.proc = c('GAM','GLM','RF','FDA','CTA','ANN','GBM','MAXNET','XGBOOST') 

##### Convert Categorical Variables to Factors ####
# If using categorical variables you cannot use MARS or SRE as modelling options
factors.to.convert <- which(names(myExpl_prediction)=="Starts_Not10",
                            names(myExpl_prediction)=="playerID")

for(j in factors.to.convert){
  myExpl_prediction[[j]] <- as.factor(myExpl_prediction[[j]])
}

##### Variable Correlation Check ####
cor_results <- cor(myExpl_prediction,
                   use = "pairwise.complete.obs")

# View the correlation matrix
print(cor_results)
corrplot(cor_results, method = "number")

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
  metric.eval = c('TSS'),
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
resp.proj.xy <- resp.proj[,1:2]

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

# Define thresholds we want to test
thresholds <- c(5,10,15,20)
stake_types <- c("fixed", "flex")
fixed_stake_amount <- 10
target_profit <- 10

#### Process each event ####
for (i in 1:length(PGA_Pred_list)) {
  # Join prediction data with actual results
  PGA_Pred_list[[i]] <- merge(
    PGA_Pred_list[[i]],
    PGA_projection[, c("eventID", "playerID", "Top20_Odds", "posn")],
    by = c("eventID", "playerID"),
    all.x = TRUE
  )
  
  # Add actual Top20 finish column
  PGA_Pred_list[[i]]$Top20 <- ifelse(PGA_Pred_list[[i]]$posn <= 20, 1, 0)
  
  # Setup odds validity
  PGA_Pred_list[[i]]$Valid_Odds <- PGA_Pred_list[[i]]$Top20_Odds > 1
  PGA_Pred_list[[i]]$Fixed_Stake <- fixed_stake_amount
  PGA_Pred_list[[i]]$Flex_Stake <- ifelse(
    PGA_Pred_list[[i]]$Valid_Odds,
    round(target_profit / (PGA_Pred_list[[i]]$Top20_Odds - 1), 2),
    0
  )
  
  # Calculate thresholds and create model score columns
  for (t in thresholds) {
    # Determine threshold value
    threshold_value <- sort(PGA_Pred_list[[i]]$Model_Score, decreasing = TRUE)[min(t, length(PGA_Pred_list[[i]]$Model_Score))]
    
    # Create model score column
    col_name <- paste0("Top", t, "_ModelScore")
    PGA_Pred_list[[i]][[col_name]] <- ifelse(PGA_Pred_list[[i]]$Model_Score >= threshold_value, 1, 0)
    
    # Calculate if model was correct
    correct_col <- paste0("Top", t, "_ModelScore_Correct")
    PGA_Pred_list[[i]][[correct_col]] <- ifelse(PGA_Pred_list[[i]]$Top20 == PGA_Pred_list[[i]][[col_name]], 1, 0)
    
    # Process each stake type
    for (stake_type in stake_types) {
      stake_col_suffix <- ifelse(stake_type == "fixed", "", "_Flex")
      stake_amount_col <- ifelse(stake_type == "fixed", "Fixed_Stake", "Flex_Stake")
      
      # Calculate stake placed
      stake_placed_col <- paste0("Top", t, "_Stake_Placed", stake_col_suffix)
      PGA_Pred_list[[i]][[stake_placed_col]] <- ifelse(
        PGA_Pred_list[[i]][[col_name]] == 1 & PGA_Pred_list[[i]]$Valid_Odds,
        PGA_Pred_list[[i]][[stake_amount_col]],
        0
      )
      
      # Calculate winnings
      winnings_col <- paste0("Top", t, "_Player_Winnings", stake_col_suffix)
      PGA_Pred_list[[i]][[winnings_col]] <- ifelse(
        PGA_Pred_list[[i]][[col_name]] == 1 & 
          PGA_Pred_list[[i]]$Valid_Odds & 
          PGA_Pred_list[[i]][[correct_col]] == 1,
        PGA_Pred_list[[i]][[stake_amount_col]] * PGA_Pred_list[[i]]$Top20_Odds,
        0
      )
      
      # Calculate profit
      profit_col <- paste0("Top", t, "_Player_Profit", stake_col_suffix)
      PGA_Pred_list[[i]][[profit_col]] <- PGA_Pred_list[[i]][[winnings_col]] - PGA_Pred_list[[i]][[stake_placed_col]]
    }
  }
}

# Create summaries
all_summaries <- data.frame()

# Function to create column name for summary metrics
col_name <- function(metric, threshold, stake_type = NULL) {
  if (is.null(stake_type)) {
    return(paste0(metric, "_top", threshold))
  } else {
    return(paste0(metric, "_", stake_type, "_top", threshold))
  }
}

# Process each event for summary statistics
for (i in 1:length(PGA_Pred_list)) {
  event_id <- unique(PGA_Pred_list[[i]]$eventID)[1]
  valid_data <- PGA_Pred_list[[i]][PGA_Pred_list[[i]]$Valid_Odds, ]
  
  # Initialize event summary with event ID
  event_summary <- data.frame(eventID = event_id)
  
  # Calculate summaries for each threshold
  for (t in thresholds) {
    # Base metrics
    model_score_col <- paste0("Top", t, "_ModelScore")
    correct_col <- paste0("Top", t, "_ModelScore_Correct")
    
    # Calculate betting totals and success rates
    total_bets <- sum(valid_data[[model_score_col]] == 1)
    successful_bets <- sum(valid_data[[model_score_col]] == 1 & valid_data[[correct_col]] == 1)
    success_rate <- ifelse(total_bets > 0, round(successful_bets / total_bets * 100, 2), 0)
    
    # Add to summary
    event_summary[[col_name("bets", t)]] <- total_bets
    event_summary[[col_name("successful", t)]] <- successful_bets
    event_summary[[col_name("success_rate", t)]] <- success_rate
    
    # Calculate for each stake type
    for (stake_type in stake_types) {
      stake_suffix <- ifelse(stake_type == "fixed", "", "_Flex")
      
      # Get column names
      stake_col <- paste0("Top", t, "_Stake_Placed", stake_suffix)
      winnings_col <- paste0("Top", t, "_Player_Winnings", stake_suffix)
      profit_col <- paste0("Top", t, "_Player_Profit", stake_suffix)
      
      # Calculate totals
      total_stake <- sum(valid_data[[stake_col]], na.rm = TRUE)
      total_winnings <- sum(valid_data[[winnings_col]], na.rm = TRUE)
      total_profit <- sum(valid_data[[profit_col]], na.rm = TRUE)
      roi <- ifelse(total_stake > 0, round(total_profit / total_stake * 100, 2), 0)
      
      # Add to summary
      event_summary[[col_name("stake", t, stake_type)]] <- round(total_stake, 2)
      event_summary[[col_name("winnings", t, stake_type)]] <- round(total_winnings, 2)
      event_summary[[col_name("profit", t, stake_type)]] <- round(total_profit, 2)
      event_summary[[col_name("roi", t, stake_type)]] <- roi
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

# Create separate sheets for each TopX in the summary workbook
for (t in thresholds) {
  # Create sheet name
  sheet_name <- paste0("Top", t, "_Summary")
  
  # Filter columns relevant to this threshold
  prefix <- paste0("_top", t)
  fixed_prefix <- paste0("_fixed_top", t)
  flex_prefix <- paste0("_flex_top", t)
  
  # Get column names for this threshold
  threshold_cols <- c(
    "eventID",
    names(all_summaries)[grepl(prefix, names(all_summaries)) & !grepl("fixed|flex", names(all_summaries))],
    names(all_summaries)[grepl(fixed_prefix, names(all_summaries))],
    names(all_summaries)[grepl(flex_prefix, names(all_summaries))]
  )
  
  # Create summary dataframe for this threshold
  threshold_summary <- all_summaries[, threshold_cols]
  
  # Add worksheet and write data
  addWorksheet(summary_wb, sheet_name)
  writeData(summary_wb, sheet_name, threshold_summary)
}

# Extract the key columns we want to include for every sheet
base_columns <- c("eventID", "playerID", "Model_Score", "Top20_Odds", "posn", "Top20")

# Create consolidated bet data for each threshold in the bets workbook
player_bets_by_threshold <- list()
for (t in thresholds) {
  # Create sheet name
  sheet_name <- paste0("Top", t, "_Bets")
  
  # Collect columns specific to this threshold
  threshold_columns <- c(
    paste0("Top", t, "_ModelScore"),
    paste0("Top", t, "_ModelScore_Correct"),
    paste0("Top", t, "_Stake_Placed"),
    paste0("Top", t, "_Player_Winnings"),
    paste0("Top", t, "_Player_Profit"),
    paste0("Top", t, "_Stake_Placed_Flex"),
    paste0("Top", t, "_Player_Winnings_Flex"),
    paste0("Top", t, "_Player_Profit_Flex")
  )
  
  # Columns to extract
  columns_to_extract <- c(base_columns, threshold_columns)
  
  # Create empty dataframe for this threshold
  player_bets_by_threshold[[t]] <- data.frame()
  
  # Extract data from each event and filter to only include actual bets
  for (i in 1:length(PGA_Pred_list)) {
    # Get data for this event
    event_data <- PGA_Pred_list[[i]]
    
    # Filter to only include players with Valid_Odds and where a ModelScore = 1 (actual bets placed)
    model_score_col <- paste0("Top", t, "_ModelScore")
    bet_data <- event_data[event_data$Valid_Odds & event_data[[model_score_col]] == 1, columns_to_extract]
    
    # Add to combined dataframe if there are any bets
    if (nrow(bet_data) > 0) {
      player_bets_by_threshold[[t]] <- rbind(player_bets_by_threshold[[t]], bet_data)
    }
  }
  
  # Add worksheet and write data if we have any bets for this threshold
  if (nrow(player_bets_by_threshold[[t]]) > 0) {
    addWorksheet(bets_wb, sheet_name)
    writeData(bets_wb, sheet_name, player_bets_by_threshold[[t]])
  }
}

# Add sheet with all players and predictions (not just bets) to bets workbook
all_players_data <- do.call(rbind, PGA_Pred_list)
addWorksheet(bets_wb, "All_Players_Data")
writeData(bets_wb, "All_Players_Data", all_players_data)

# Save the workbooks
summary_excel_file_path <- paste0("./Betting Simulations/Top 20/PGA_Event_Betting_Outcomes_Model", run, ".xlsx")
bets_excel_file_path <- paste0("./Betting Simulations/Top 20/PGA_Player_Betting_Outcomes_Model", run, ".xlsx")

saveWorkbook(summary_wb, summary_excel_file_path, overwrite = TRUE)
saveWorkbook(bets_wb, bets_excel_file_path, overwrite = TRUE)
