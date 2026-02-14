##
##
##
## cluster.name = county
## sensitive = sensitivity to be reached (95)
## model.predictions = the median or wmean raster file for the ensemble model (needs some thought for the multi-PA version)
## model.PAs = not a great name.  It is the myBiomodData object
library(raster)
cutoff.finder.core<-function(cluster.name,sensitive,model.predictions,model.PAs)
{
  if(missing(model.PAs))
  {
    orig.PAs<-F
    stop("Original Pseudoabsence notfound \n ")
  }
  else
  {
    #check the model.PAs is the correct format (should probably do for all the others but hey)
    if(class(model.PAs)[1]=="BIOMOD.formated.data.PA")
    {
      orig.PAs<-T
      cat("Original Pseudoabsence found\n ")
      #model.PAs.data<-model.PAs@coord[grep("pa",rownames(model.PAs@coord)),]
      
      model.absencePA.data<-model.PAs@coord[which(model.PAs@data.species %in% c(0,NA)),]
    }
    else
    {
      orig.PAs<-F
      cat("Original Pseudoabsence found but not correct format, using Pseudo instead\n ")
    }
  }
  
  #Total number of records (Pres/Abs/Pseudo)
  no.of.records<-length(model.PAs@data.species)
  no.of.presence<-length(which(model.PAs@data.species==1))
  no.of.absence<-length(which(model.PAs@data.species==0))
  no.of.pseudo<-length(which(is.na(model.PAs@data.species)))
  
  
  #Create data frame with coords and species (Pres/Abs/Psuedo)
  gcn.records<-cbind(model.PAs@coord,model.PAs@data.species)
  names(gcn.records)<-c("EASTING","NORTHING","PresAbs")
  
  #The above has pseudo set as NA, change to NA (new dataframe for the sake of it?!?)
  gcn.records.pseudo.to.absence<-gcn.records
  gcn.records.pseudo.to.absence[which(is.na(gcn.records.pseudo.to.absence$PresAbs)),"PresAbs"]<- -1
  if(class(model.predictions)=="SpatRaster"){model.predictions<-raster(model.predictions)}
  
  #rasterize presabs
  gcn.ras<-rasterize(gcn.records.pseudo.to.absence[,c("EASTING","NORTHING")],model.predictions,field = gcn.records.pseudo.to.absence$PresAbs)
  #get the predictions in these grid cells from the ensemble prediction
  predictions.mask.ras<-raster::mask(model.predictions,gcn.ras)
  #create a table with the presence/absence/pseudo absence locations 
  gcn.table<-rasterToPoints(gcn.ras)
  #create a list of the predictions for the data points
  predictions.table<-rasterToPoints(predictions.mask.ras)
  #create a table with X Y pres/abs/pseudo/prediction
  #test.table<-as.data.frame(cbind(gcn.table,predictions.table[,3]))
  #colnames(test.table)<-c("EASTING","NORTHING","PresAbs","Prediction")
  
  gcn.table.df<-as.data.frame(gcn.table)
  predictions.table.df<-as.data.frame(predictions.table)
  colnames(gcn.table.df)<-c("EASTING","NORTHING","type")
  colnames(predictions.table.df)<-c("EASTING","NORTHING","prediction")
  test.table<-gcn.table.df %>% inner_join( predictions.table.df, 
                         by=c('EASTING'='EASTING', 'NORTHING'='NORTHING'))
  colnames(test.table)<-c("EASTING","NORTHING","PresAbs","Prediction")
  
  no.of.records<-nrow(test.table)
  no.of.presence<-nrow(test.table[which(test.table$PresAbs==1),])
  no.of.absence<-nrow(test.table[which(test.table$PresAbs==0),])
  no.of.pseudo<-nrow(test.table[which(test.table$PresAbs== -1),])
  
  #Now we're going to run through from 1000 to 0 in 0.5 intervals - dropping the threshold (prediction 1 to 0)
  #At each point we check to see how many presences we are correctly predicting based on the threshold and until we find 95%
  #
  i=1
  cutoff<-1000
  pres.table<-test.table[which(test.table$PresAbs==1),]
  #get number of presences (or reuse!!!) to test against the number predicted at each stage
  presences<-no.of.presence
  #need to store the decreasing model sensitivity 
  model.sensitivity<-rolling.sensitivity<-0
  while(model.sensitivity<sensitive)
  {
    tmp.table<-pres.table
    tmp.table<-tmp.table[which(tmp.table$Prediction>cutoff),]
    predicted.presences<-nrow(tmp.table)
    if(predicted.presences>0)
    {
      rolling.sensitivity<-(predicted.presences/presences*100)
      if(rolling.sensitivity>model.sensitivity)
      {
        model.sensitivity<-rolling.sensitivity
        #cat(cutoff,model.sensitivity,"\n")
      }
    }
    #drop .5 and restart
    cutoff<-cutoff-0.5
    #print(rolling.sensitivity)
  }
  
  
  cutoff<-cutoff+0.5 #dropped .5 at the end of the while loop - could be done in the if statement to avoid this!
  cat("cutoff:",cutoff,"sensitivity:",model.sensitivity,"\n")
  
  #pseudo absence calc
  mod.predictions<-model.predictions
  
  
  #set map to be presence/absence based on new 95% cutoff
  gcn.presence.map<-model.predictions
  gcn.presence.map[which(gcn.presence.map[]<cutoff)]<-0
  gcn.presence.map[which(gcn.presence.map[]>=cutoff)]<-1
  
  
  threshold<-cutoff#why did I do this :)
  model.specificities<-correct.pseudoabsences<-mappable.pseudo<-NULL
  
  #cat("pseudoabsence selection\n")
  model.absencePA.data<-test.table[which(test.table$PresAbs %in% c(0,-1)),]
  # get the raster cell id from the xy coord of absence and pseudoabs
  samp<-cellFromXY(gcn.presence.map,model.absencePA.data[,c(1,2)])
  
  # and their values for pres/abs/pseudo
  sampdata <- gcn.presence.map[samp]
  # and the prediction scores
  sampdataforAUC<-model.predictions[samp]
  
  total.sample.size<-length(sampdata)  # also above as ........ no.of.records
  
  #predpres<-model.predictions[which(gcn.ras[]==1)]/1000
  #predabs<-sampdataforAUC/1000
  #get the confusion matrix values (true presence, true absence, false presence, false absence)
  abcd<-get_abcd(model.PAs,threshold,model.predictions)
  #calculate the TSS and other scores
  scoring<-get_scoring(abcd)
  cat("Final ",i,": Threshold:",threshold,"; Sensitivity:",scoring["vH"],"; MEAN Specificity:",scoring["vS"],"\n",sep="")  
  
  return(c(threshold=threshold,
           sensitivity95=as.numeric(scoring["vH"]),
           specificity95=as.numeric(scoring["vS"]),
           AUC95=NA,#auc_obj_95$auc[1], # not doing AUC anymore
           TSS95=as.numeric(scoring["TSS"]),
           ORSS95=as.numeric(scoring["ORSS"]),
           SEDI95=as.numeric(scoring["SEDI"]),
           bias95=as.numeric(scoring["bias"]),
           prevalance95=as.numeric(scoring["prevalance"])))
  
  
}

cutoff.finder.specificity <- function(cluster.name, specific, model.predictions, model.PAs) {
  # Input validation
  if(missing(model.PAs)) {
    stop("Original Pseudoabsence not found")
  }
  
  if(class(model.PAs)[1] != "BIOMOD.formated.data.PA") {
    stop("model.PAs is not in the correct format")
  }
  
  # Convert SpatRaster to raster if necessary
  if(class(model.predictions) == "SpatRaster") {
    model.predictions <- raster(model.predictions)
  }
  
  # Prepare data
  gcn.records <- cbind(model.PAs@coord, model.PAs@data.species)
  names(gcn.records) <- c("EASTING", "NORTHING", "PresAbs")
  
  # Initialize variables
  cutoff <- 0  # Start with a low cutoff (high sensitivity)
  target_sensitivity <- 100  # Start with 100% sensitivity
  model.specificity <- 0
  max_iterations <- 2000
  
  cat("Starting cutoff search. Target specificity:", specific, "\n")
  
  for(i in 1:max_iterations) {
    abcd <- get_abcd(model.PAs, cutoff, model.predictions)
    scoring <- get_scoring(abcd)
    model.sensitivity <- scoring["vH"] * 100
    model.specificity <- scoring["vS"] * 100
    
    cat("Iteration", i, ": Cutoff =", cutoff, ", Target Sensitivity =", target_sensitivity, 
        ", Actual Sensitivity =", model.sensitivity, ", Specificity =", model.specificity, "\n")
    
    if(model.specificity >= specific) {
      cat("Target specificity reached.\n")
      break
    }
    
    target_sensitivity <- target_sensitivity - 0.5
    cutoff <- quantile(model.predictions[], probs = (100 - target_sensitivity) / 100, na.rm = TRUE)
    
    if(target_sensitivity < 0) {
      warning("Target sensitivity reached 0. Desired specificity may not be achievable.")
      break
    }
  }
  
  if(i == max_iterations) {
    warning("Maximum iterations reached. Desired specificity may not be achievable.")
  }
  
  cat("Final cutoff:", cutoff, "Final specificity:", model.specificity, "Final sensitivity:", model.sensitivity, "\n")
  
  # Calculate final metrics
  abcd <- get_abcd(model.PAs, cutoff, model.predictions)
  scoring <- get_scoring(abcd)
  
  cat("Final results: Threshold:", cutoff, "; Specificity:", scoring["vS"] * 100, "; Sensitivity:", scoring["vH"] * 100, "\n")
  
  return(c(threshold = cutoff,
           specificity = scoring["vS"] * 100,
           sensitivity = scoring["vH"] * 100,
           TSS = scoring["TSS"],
           ORSS = scoring["ORSS"],
           SEDI = scoring["SEDI"],
           bias = scoring["bias"],
           prevalence = scoring["prevalance"]))
}

get_abcd<-function(model.data,cutoff,predictions)
{
  #get the cells for the data points (pres/abs/pseudoabsence)
  cells<-cellFromXY(predictions,model.data@coord)
  getPredictions<-predictions[cells]
  presence<-getPredictions[which(model.data@data.species==1)]
  absence<-getPredictions[which(model.data@data.species %in% c(0,NA))]
  vA<-length(which(presence>=cutoff))#present and predicted present - true presence
  vB<-length(which(absence>=cutoff))#absent but predicted present - false presence
  vC<-length(which(presence<cutoff))#present but predict absent - false absence
  vD<-length(which(absence<cutoff))#absent and predicted absent - true absence
  return(c(a=vA,b=vB,c=vC,d=vD))
  
}

get_scoring<-function(abcd)
{
  # aA<-global.model.sensitivity#Presence and predicted present
  # bB<-100-mean(model.specificities)#Absent but predicted present
  # cC<-100-global.model.sensitivity#Presence but predicted absent
  # dD<-mean(model.specificities)#Absent and predicted absent
  # tT<-aA/(2*(aA+cC))
  # uU<-dD/(2*(bB+dD))
  # aAUC<-tT+uU
  va<-as.numeric(abcd["a"])
  vb<-as.numeric(abcd["b"])
  vc<-as.numeric(abcd["c"])
  vd<-as.numeric(abcd["d"])
  vH<-va/(va+vc) # also known as sensitivity
  vF<-vb/(vb+vd) # 1-specificity
  vS<-abs(vF-1) # specificity
  vbias<-(va+vb)/(va+vc) # frequency bias
  vprevalance<-(va+vc)/(va+vb+vc+vd) # also known as base rate
  
  rTSS<-vH-vF
  rORSS<-((va*vd)-(vb*vc))/((va*vd)+(vb*vc))
  rSEDI<-(log(vF)-log(vH)-log(1-vF)+log(1-vH))/(log(vF)+log(vH)+log(1-vF)+log(1-vH))
  
  return(c(vH=vH,vF=vF,vS=vS,TSS=rTSS,ORSS=rORSS,SEDI=rSEDI,bias=vbias,prevalance=vprevalance))
}


cutoff.finder.speed<-function(cluster.name,sensitive,model.predictions,reps,exclude_absences,gcn.records,additional.records,model.PAs)
{
  if(missing(model.PAs))
  {
    orig.PAs<-F
    cat("Original Pseudoabsence not used\n ")
  }
  else
  {
    if(class(model.PAs)[1]=="BIOMOD.formated.data.PA")
    {
      orig.PAs<-T
      cat("Original Pseudoabsence found\n ")
      #model.PAs.data<-model.PAs@coord[grep("pa",rownames(model.PAs@coord)),]
      
      model.absencePA.data<-model.PAs@coord[which(model.PAs@data.species %in% c(0,NA)),]
    }
    else
    {
      orig.PAs<-F
      cat("Original Pseudoabsence found but not correct format, using Pseudo instead\n ")
    }
    
  }
  
  if(missing(gcn.records)&missing(additional.records))
  {
    if(class(model.PAs)[1]=="BIOMOD.formated.data.PA")
    {
      #cutoff.finder.speed(county,95,r2.bkp,reps,PresenceOnly.bkp,model.PAs=myBiomodData.bkp)
      model.PresAbs.data<-as.data.frame(model.PAs@coord[which(model.PAs@data.species %in% c(0,1)),])
      model.PresAbs.data<-cbind(model.PresAbs.data,model.PAs@data.species[which(model.PAs@data.species %in% c(0,1))])
      model.PresAbs.data<-cbind(model.PresAbs.data,landscapeType=0)
      gcn.records.bkp<-model.PresAbs.data
      names(gcn.records.bkp)<-c("EASTING","NORTHING","PresAbs","landscapeType")
      gcn.records.bkp<-gcn.records.bkp[with(gcn.records.bkp,order(EASTING,NORTHING,-PresAbs)),]
      gcn.records.bkp<-gcn.records.bkp[!duplicated(gcn.records.bkp[,c("EASTING","NORTHING")]),]
      
    }
    else
    {
      gcn.records.bkp<-read.csv(paste0(getwd(),"/data/",cluster.name,"eDNA.csv"))
      gcn.additional<-read.csv(paste0(getwd(),"/data/",cluster.name,"_required_license_data.csv"))
      names(gcn.additional)[which(names(gcn.additional)=="landscapeCode")]<-"landscapeType"
      gcn.records.bkp<-rbind(gcn.records.bkp,gcn.additional[,c("EASTING","NORTHING","PresAbs","landscapeType")])
      
    }
    
  }
  else if(!missing("gcn.records")&!missing("additional.records"))
  {
    #these filenames can include the full path
    cat("\n",gcn.records,"\n",additional.records)
    if(substr(gcn.records,2,2)==":")
    {
      gcn.records.bkp<-read.csv(gcn.records)
    }
    else
    {
      gcn.records.bkp<-read.csv(paste0(getwd(),"/data/",gcn.records,".csv"))
    }
    if(substr(additional.records,2,2)==":")
    {
      gcn.additional<-read.csv(additional.records)
    }
    else
    {
      gcn.additional<-read.csv(paste0(getwd(),"/data/",additional.records,".csv"))
    }
    if(length(names(gcn.additional)[which(names(gcn.additional)=="Year")])==0)
    {
      print("No year information in additional data, assuming all valid!!!!")
      gcn.additional$Year<-2019
    }
    if(length(names(gcn.records.bkp)[which(names(gcn.records.bkp)=="Year")])==0)
    {
      print("No year information in gcn main data data, assuming all valid!!!!")
      gcn.records.bkp$Year<-2019
    }
    if(length(names(gcn.additional)[which(names(gcn.additional)=="GridRefRes")])==0)
    {
      print("No Grid Reference Resolution information in additional data, assuming all <=25m!!!!")
      gcn.additional$GridRefRes<-10
    }
    if(length(names(gcn.records.bkp)[which(names(gcn.records.bkp)=="GridRefRes")])==0)
    {
      print("No Grid Reference Resolution information in gcn main data data, assuming all <=25m!!!!")
      gcn.records.bkp$GridRefRes<-10
    }
    
    
    gcn.records.bkp<-rbind(gcn.records.bkp[,c("EASTING","NORTHING","PresAbs","Year","GridRefRes")],gcn.additional[,c("EASTING","NORTHING","PresAbs","Year","GridRefRes")])
    gcn.records.bkp<-gcn.records.bkp[which(gcn.records.bkp$PresAbs > -1 &as.numeric(gcn.records.bkp$Year)>2013 & gcn.records.bkp$GridRefRes<100),]
    #order the df so the removal of duplicates removes the oldest and more importantly retains the presence over the absence
    gcn.records.bkp<-gcn.records.bkp[with(gcn.records.bkp,order(EASTING,NORTHING,-PresAbs,-Year)),]
    gcn.records.bkp<-gcn.records.bkp[!duplicated(gcn.records.bkp[,c("EASTING","NORTHING")]),]
    
  }
  else
  {
    stop("Both gcn.records and additional.records are required" )
  }
  
  #nrow(gcn.records)
  invalid.pts <- which(is.na(extract(model.predictions, gcn.records.bkp[,1:2])))
  if(length(invalid.pts)>0)
  {
    gcn.records.bkp<-gcn.records.bkp[-invalid.pts,]
  }
  nrow(gcn.records.bkp)
  no.of.records<-nrow(gcn.records.bkp)
  no.of.presence<-nrow(gcn.records.bkp[which(gcn.records.bkp$PresAbs==1),])
  
  write.csv(gcn.records.bkp,paste0(getwd(),"/data/data_from_cutoff_finder.csv"),row.names = F)
  
  gcn.records<-gcn.records.bkp
  #rasterize presabs
  gcn.ras<-rasterize(gcn.records[,c("EASTING","NORTHING")],model.predictions,field = gcn.records$PresAbs)
  #get the predictions in these grid cells
  predictions.mask.ras<-raster::mask(model.predictions,gcn.ras)
  #create a table with them
  gcn.table<-rasterToPoints(gcn.ras)
  predictions.table<-rasterToPoints(predictions.mask.ras)
  test.table<-as.data.frame(cbind(gcn.table,predictions.table[,3]))
  colnames(test.table)<-c("EASTING","NORTHING","PresAbs","Prediction")
  i=1
  cutoff<-1000
  pres.table<-test.table[-which(test.table$PresAbs==0),]
  presences<-nrow(pres.table)
  model.sensitivity<-rolling.sensitivity<-0
  true.presence<-false.presence<-false.absence<-true.presence<-0
  while(model.sensitivity<sensitive)
  {
    tmp.table<-pres.table
    tmp.table<-tmp.table[which(tmp.table$Prediction>cutoff),]
    predicted.presences<-nrow(tmp.table)
    if(predicted.presences>0)
    {
      rolling.sensitivity<-(predicted.presences/presences*100)
      if(rolling.sensitivity>model.sensitivity)
      {
        model.sensitivity<-rolling.sensitivity
        #cat(cutoff,model.sensitivity,"\n")
      }
    }
    cutoff<-cutoff-0.5
  }
  
  
  cutoff<-cutoff+0.5 #while loop end reset
  cat("cutoff:",cutoff,"sensitivity:",model.sensitivity,"\n")
  #pseudo absence calc
  mod.predictions<-model.predictions
  gcn.presence.map<-model.predictions
  gcn.presence.map[which(gcn.presence.map[]<cutoff)]<-0
  gcn.presence.map[which(gcn.presence.map[]>=cutoff)]<-1
  
  
  threshold<-cutoff
  model.specificities<-correct.pseudoabsences<-mappable.pseudo<-NULL
  
  if(orig.PAs)
  {
    #cat("pseudoabsence selection\n")
    
    # get the cell id from the xy coord of absence and pseudoabs
    samp<-cellFromXY(gcn.presence.map,model.absencePA.data[,c(1,2)])
    
    # and their values
    sampdata <- gcn.presence.map[samp]
    sampdataforAUC<-model.predictions[samp]
    # and their location coordinates
    #samplocs <- xyFromCell(raster, samp)
    
    correct.absences<-length(sampdata[which(sampdata==0)])
    
    total.sample.size<-length(sampdata)
    model.specificity<-(correct.absences/total.sample.size)*100
    model.specificities<-c(model.specificities,model.specificity)
    
  }
  else
  {
    runs<-10 #pseudo absence run number
    absence.ras<-gcn.ras
    absence.ras[which(absence.ras[]==1)]<-NA
    num.true.absence<-length(absence.ras[which(!is.na(absence.ras[]))])
    #this gets the prediction of absence versus actual absence
    absence.mask.ras<-raster::mask(gcn.presence.map,absence.ras)
    predicted.num.true.absence<-length(absence.mask.ras[which(absence.mask.ras[]==0)])
    for(i in 1:runs)
    {
      #if(is.null(myBiomodData.bkp))
      if(1==1)
      {
        #set the PAs based on the reps (number of PAs based on GCN records x reps)
        #cat("pseudoabsence selection\n")
        pseudoabsence.points<-no.of.records*reps#number of pseudo absence points
        notna <- which(!is.na(values(gcn.presence.map)))
        
        # grab random cell index numbers at random points
        samp <- sample(notna, pseudoabsence.points, replace = FALSE)
        
        # and their values
        sampdata <- gcn.presence.map[samp]
        
        # and their location coordinates
        #samplocs <- xyFromCell(raster, samp)
        
        pseudoabsence.num.absence<-length(sampdata[which(sampdata==0)])
        
        correct.absences<-pseudoabsence.num.absence+predicted.num.true.absence
        total.sample.size<-pseudoabsence.points+num.true.absence
        model.specificity<-(correct.absences/total.sample.size)*100
        model.specificities<-c(model.specificities,model.specificity)
        
      }
      else
      {
        #get pseudo absences from myBiomodData.bkp 
      }
      cat("Run   ",i,": Threshold:",threshold,"; Sensitivity:",model.sensitivity,"; Specificity:",model.specificity,"\n",sep="")  
    }
  }
  
  
  global.model.sensitivity<-model.sensitivity
  cat("Final ",i,": Threshold:",threshold,"; Sensitivity:",global.model.sensitivity,"; MEAN Specificity:",mean(model.specificities),"\n",sep="")  
  
  #obs<-c(rep(1,no.of.presence),rep(0,pseudoabsence.points))
  
  obs<-c(rep(1,no.of.presence),rep(0,total.sample.size))
  
  predpres<-model.predictions[which(gcn.ras[]==1)]/1000
  
  if(orig.PAs)
  {
    predabs<-sampdataforAUC/1000
    abcd<-get_abcd(model.PAs,threshold,model.predictions)
    scoring<-get_scoring(abcd)
    return(c(threshold=threshold,
             sensitivity95=global.model.sensitivity,
             specificity95=mean(model.specificities),
             AUC95=NA,#auc_obj_95$auc[1], # not doing AUC anymore
             TSS95=scoring["TSS"],
             ORSS95=scoring["ORSS"],
             SEDI95=scoring["SEDI"],
             bias95=scoring["bias"],
             prevalance95=scoring["prevalance"]))
    
  }
  else
  {
    predabs<-sampleRandom(model.predictions,total.sample.size)/1000  
    tss_95<-(global.model.sensitivity/100)+((mean(model.specificities)/100) - 1)
    return(c(threshold=threshold,
             sensitivity95=global.model.sensitivity,
             specificity95=mean(model.specificities),
             AUC95=NA,#auc_obj_95$auc[1], # not doing AUC anymore
             TSS95=tss_95,
             ORSS95=scoring["ORSS"],
             SEDI95=scoring["SEDI"],
             bias95=scoring["bias"],
             prevalance95=scoring["prevalance"]))
    
  }
  
  
  
  #table.for.ROC<-data.frame(category=c(rep(1,no.of.presence),rep(0,pseudoabsence.points)),prediction=c(rep(1,threshold.sensitivity),
  #                                                                                                     rep(0,no.of.presence-threshold.sensitivity),
  #                                                                                                     rep(0,round(pseudoabsence.points*(mean(model.specificities)/100))),
  #                                                                                          rep(1,round(pseudoabsence.points*(abs(mean(model.specificities)-100)/100)))))
  # aA<-global.model.sensitivity#Presence and predicted present
  # bB<-100-mean(model.specificities)#Absent but predicted present
  # cC<-100-global.model.sensitivity#Presence but predicted absent
  # dD<-mean(model.specificities)#Absent and predicted absent
  # tT<-aA/(2*(aA+cC))
  # uU<-dD/(2*(bB+dD))
  # aAUC<-tT+uU
  
  
  #auc_obj_95<-pROC::roc(factor(obs),c(predpres,predabs))
  #plot(auc_obj)
  #cat("AUC: ",auc(roc_obj),"\n")
  #unique(c(predpres,predabs))
  #roc_df <- data.frame(TPR=rev(roc_obj$sensitivities),
  #                     TPR=rev(1-roc_obj$specificities),
  #                     labels=roc_obj$response,
  #                     scores=roc_obj$predictor)
  #plot(roc_obj,auc.polygon=T,print.auc=T,grid=c(0.1,0.1),max.auc.polygon=T,auc.polygon.col="lightblue")
  #power.roc.test(roc_obj,method="obuchowski")
  
  #library(PresenceAbsence)
  #df<-data.frame(rownum=seq(1,length(obs),1),observations=obs,predictions=c(predpres,predabs))
  #auc_obj2<-PresenceAbsence::auc(df)
  
  
  
  
  
}
