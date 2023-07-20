# InterDemo
# script for training various ml models of civil war intervention
# this is take two, creating a better workflow and standard procedure for preprocessing etc.

library(tidyverse)
library(tree)
library(randomForest)
library(ggplot2)
library(nnet)
library(sandwich)
library(lmtest)
library(stargazer)
library(mlogit)
library(scales)
library(expss)
library(car)
library(caret)
library(glmnet)
library(skimr)
library(RANN)
library(kernlab)
library(xgboost)
library(foreach)
library(doParallel)
library(MLmetrics)
library(countrycode)
library(grid)
library(gridExtra)
library(yardstick)
#library(fastAdaboost) #not available anymore, but need it for Adaboost algo

# NOTE:
# add suite of interactions, run through lasso, cut, then RF again
# bag svm: reduced space svm, iteration ???
# need to reduce dimensionality as well as add more features

# NOTES ON RE-DO:
# other pre-processing required, like transformation

# DiagPlot function code for model critique --------------------------------------------------

# taken from Mahmood git. had to do this because package isn't on cran and github install isn't working

DiagPlot <- function(f, y, labels, worstN=10, size_adjust=0,right_margin=7,top_margin=1.5,label_spacing=10,lab_adjust=.5,text_size=10,bw=FALSE,title="Model Diagnostic Plot") {
  
  #################
  #Begin Function
  #################
  
  data <- data.frame(f=f, y=y, labels=labels)
  pdata <- data %>% mutate(y_minus_f=y-f) %>% arrange(f) %>% mutate(forecastOrder = row_number())
  #still need to label worstN
  pdata <- pdata %>% group_by(y) %>% arrange(desc(abs(y_minus_f))) %>% mutate(label_worst=ifelse(row_number()<=worstN, as.character(labels), " "))
  #need to create var for absolute errors
  pdata<-pdata%>%mutate(abserr=abs(y_minus_f))
  #create indicator for worst values
  pdata <- pdata %>% group_by(y) %>% arrange(desc(abs(y_minus_f))) %>% mutate(isworstn=ifelse(row_number()<=worstN, 1, 0))
  #for coloring
  pdata <- pdata %>% mutate(coloring=
                              ifelse(y==1 & isworstn==1, '1w',
                                     ifelse(y==0 & isworstn==1, '0w',
                                            ifelse(y==1 & isworstn==0, '1',
                                                   '0'))))
  #arrange data for plotting
  pdata<-pdata%>%arrange(forecastOrder)
  N=nrow(pdata)
  labbuffer=(nchar(N)-3)*.3
  #Colors for use
  yblue=ifelse(bw==F,'#0862ca','#8b8b8b')
  ybluemarg=ifelse(bw==F,yblue,"#989898")
  ybluelite=ifelse(bw==F,'#cddff4','#d8d8d8')
  ybluelitest=ifelse(bw==F,'#f0f5fb','#f2f2f2')
  yred=ifelse(bw==F,'#fd1205','#000000')
  yredmarg=ifelse(bw==F,yred,yred)
  yredlite=ifelse(bw==F,'#fecfdc','#999999')
  yredlitest=ifelse(bw==F,'#fef0f4','#e5e5e5')
  boolcolors<-as.character(c(
    '1w'=ybluelite, #very light blue
    '0w'=yblue, #bold blue
    '1'=yredlite, #very light red
    '0'=yred)) #bold red
  boolscale<-scale_color_manual(name='coloring',values=boolcolors)
  ###################
  #initialize plots.
  #	Object "o2" contains the full plot we care about,
  #		minus the lines & labels. 
  #	Object "margx" is the marginal on the x axis of f|y=0 & f|y=1
  ###################
  o1 <- ggplot(pdata, aes(x=f,y=forecastOrder,group=y, color=as.factor(coloring)))+boolscale
  o2 <- o1 + geom_point(aes(alpha=(isworstn)))  +geom_rug(sides="r")+xlim(c(0,1))+ylim(c(0,N))+theme_bw()+theme(panel.grid.major=element_line(colour='grey'),panel.grid.minor=element_line(colour='grey'),panel.grid.major.y=element_blank(),panel.grid.minor.y=element_blank(),panel.grid.minor.x=element_blank(),axis.text.x=element_blank(),axis.ticks.x=element_blank(),axis.title.x=element_blank(),legend.position='none',plot.margin=unit(c(top_margin,right_margin,-.2,1),"lines")) +labs(y='Observation (ordered by f)')+boolscale
  margx<-ggplot(pdata,aes(f,fill=factor(y)))+geom_density(alpha=.4)+scale_fill_manual(values=c(yblue,yredmarg))+xlim(c(0,1))+labs(x='Forecast Value')+theme_bw()+theme(panel.grid.minor=element_blank(),panel.grid.major=element_blank(),axis.title.y=element_blank(),axis.text.y=element_blank(),axis.ticks.y=element_blank(),legend.position="none",plot.margin=unit(c(0,right_margin,0.2,3.35+labbuffer),"lines"))
  
  ###################
  #Lines and Labels
  ###################	
  z<-o2
  count0=0
  count1=0
  #yblue<-ifelse(bw==F,'blue',yblue)
  #yred<-ifelse(bw==F,'red',yred)
  for (i in 1:length(pdata$label_worst)) {
    
    ################################
    #Prepare to position labels
    ################################	
    text_spacing<-label_spacing
    
    labeltext<-pdata$label_worst[i]
    if(labeltext == ' '){
      next
    }
    obsy=pdata$y[i]
    if(obsy==0){
      count0<-count0+text_spacing
    }
    if(obsy==1){
      count1<-count1+text_spacing
    }
    if(count1==text_spacing){ 
      y1init=pdata$forecastOrder[i]
    }
    if(count0==text_spacing){
      y0init=pdata$forecastOrder[i]
    }
    
    fpos<-pdata$f[i]
    ##############################
    #Set the parameters for labels
    ##############################
    ycolor<-ifelse(obsy==0,yblue,yred)
    ypos_text<-ifelse(obsy==0,
                      (y0init+(count0-text_spacing+400)), # ugly but necessary modification here because false negatives and false positives are so close
                      (y1init+(count1-400-text_spacing*worstN)) # adding a multiplier here just to get cleaner separation
    )
    ifelse(pdata$forecastOrder[i]>ypos_text,LineSlope<-c(1,0),LineSlope<-c(0,1))
    labjust_left=1.1
    labjust_right=labjust_left+lab_adjust
    
    ###############################
    #Create the labels on plot
    ###############################
    current<-
      z+
      annotation_custom(
        grob=textGrob(label=labeltext,
                      gp=gpar(fontsize=text_size,col=ycolor)),
        ymin=ypos_text,
        ymax=ypos_text,
        xmin=labjust_left,
        xmax=labjust_right
      )+
      annotation_custom(
        grob=linesGrob(
          x=c(1,labjust_left),
          y=LineSlope,
          gp=gpar(col=ycolor)
        ),
        ymin=
          ifelse(
            pdata$forecastOrder[i]<=ypos_text,
            pdata$forecastOrder[i],
            ypos_text),
        ymax=
          ifelse(
            pdata$forecastOrder[i]>ypos_text,
            pdata$forecastOrder[i],
            ypos_text)
      )+
      annotation_custom(
        grob=linesGrob(
          x=c(fpos+.05,.95),
          y=0,
          gp=gpar(col=ifelse(obsy==0,ybluelitest,yredlitest))
        ),
        ymin=pdata$forecastOrder[i],
        ymax=pdata$forecastOrder[i]
      )
    z<-current
  }
  
  #Turn off clipping so we can render the plot
  gt <- ggplot_gtable(ggplot_build(z))
  gt$layout$clip[gt$layout$name == "panel"] <- "off"
  o3<-arrangeGrob(gt,margx,ncol=1,nrow=2,heights=c(4+size_adjust,1-size_adjust),top=textGrob(title,gp=gpar(fontsize=15,font=2),just='top'))
  return(o3)
}

# Trim main dataframe -----------------------------------------------------

# df.int is the main dataframe, which includes observations where T supports both sides starting in same year

# dropping irrelevant variables
df.ml <- df.int %>%
  select(., -c("rivalryno", "rivalryname", "styear", "endyear", "govsupport", "rebsupport", "govsupportyear", "rebsupportyear", "bothsides", "geo_NA", "region", "type1", "type2", "type3", 'govrec', 'rebrec')) %>%
  select(intervention, everything())
# moving DV first in df

# dropping observations with intervention on both sides in the same year
df.ml <- df.ml[df.ml$intervention!=3, ]
df.ml$intervention <- factor(df.ml$intervention,
                                        levels = c(0, 1, 2))

# # need to fix the level names cause caret can't handle the numerical values
levels(df.ml$intervention) <- c("neutral", "gov", "rebel")


# need to fix ucdponset and pol_rel

df.ml$ucdponset[is.na(df.ml$ucdponset)] <- 0
table(df.ml$ucdponset)

df.ml$pol_rel[is.na(df.ml$pol_rel)]=0
table(df.ml$pol_rel)

# Partitioning data -------------------------------------------------------

set.seed(97)

# not sure if I need to hot-one the DV
train_RowNumbers <- createDataPartition(df.ml$intervention, p=0.8, list = FALSE)

df.ml.train <- df.ml[train_RowNumbers,]

df.ml.test <- df.ml[-train_RowNumbers,]

# now pulling out identifiers into separate df
trainID <- df.ml.train %>% select(ccode1, ccode2, conflictID, year)
testID <- df.ml.test %>% select(ccode1, ccode2, conflictID, year)

df.ml.train <- subset(df.ml.train, select = -c(ccode1, ccode2, conflictID, year))
df.ml.test <- subset(df.ml.test, select = -c(ccode1, ccode2, conflictID, year))

# storing rhs and outcome for later
x = df.ml.train[, -1]
y = df.ml.train$intervention

# Descriptive statistics --------------------------------------------------

table(df.ml.train$intervention)
table(df.ml.test$intervention)

skimmed <- skim(df.ml.train)

skimmed[2:16,]

skimmed[17:41,]

skimmed[42:56,]

skimmed[57:71,]

skimmed[72:74,] # might not be correct final column number

# overall, pretty good coverage on most variables except the San-Akca ones

# Preprocessing --------------------------------------------------------------

# imputation

# picked k = 3 at random, while trying to troubleshoot an error
missing_model <- preProcess(as.data.frame(df.ml.train), method='knnImpute', k = 3)
missing_model

df.ml.train <- predict(missing_model, df.ml.train)
anyNA(df.ml.train)

# hot-ones: not needed cause only categorical variable is the DV

# other transformation: gonna transform to range, since some vars vary a lot in value range

range_model <- preProcess(as.data.frame(df.ml.train), method='range')

df.ml.train <- predict(range_model, df.ml.train)

skim(df.ml.train)
# looks good

# not sure what all these options do
featurePlot(x = df.ml.train[, 20:27],
            y = df.ml.train$intervention,
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.8)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")))

# Recursive feature elimination -------------------------------------------

set.seed(46)
options(warn=-1)

subsets <- c(58, 63, 68, 73)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=df.ml.train[, 2:74], y=df.ml.train$intervention,
                 sizes = subsets,
                 rfeControl = ctrl)
# taking a while to compute
lmProfile

rfe.predictors <- predictors(lmProfile)

rfe.cut <- df.ml.train[, which((names(df.ml.train) %in% rfe.predictors)==FALSE)] %>%
  colnames(.)
rfe.cut <- rfe.cut[-1]
# taking the DV out of the list

df.ml.train <- df.ml.train[, which((names(df.ml.train) %in% rfe.cut)==FALSE)]

# Take 1: RF model training -------------------------------------------------------

modelLookup('rf')

set.seed(111)

#model_rf = train(intervention ~ ., data=df.ml.train, method='rf')
# this is insanely slow; what's the default tune length?

# doing some hyper parameter tuning
fitControl <- trainControl(
  method = 'cv', #k-fold cross validation
  number = 5,
  savePredictions = 'final',
#  classProbs = T, # I want predictions rather than probabilities
  summaryFunction = multiClassSummary
)

model_rf = train(intervention ~ ., data=df.ml.train, method='rf', tuneLength = 5, trControl = fitControl, metric = "Mean_F1")
# using Mean_F1 as the metric here, because accuracy and kappa are not appropriate for class-imbalanced data

model_rf

fitted.rf <- predict(model_rf)

prob.rf <- predict(model_rf, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3) # forgot why I did this..

# looking at variable importance

varimp_rf <- varImp(model_rf)

plot(varimp_rf, main = "Variable importance with Random Forest")


# Take 1: Elastic net training ---------------------------------------------------

## elnet (glmnet)

model_elnet = train(intervention ~ ., 
                    data=df.ml.train, 
                    method='glmnet', 
#                    tuneLength = 5,
                    tuneGrid = expand.grid(
                      alpha = 0:1, 
                      lambda = seq(0, 15, by = 0.01)), #upping penalty here to see if it improves performance
                    maxit = 100000,
                    trControl = fitControl, metric = "Mean_F1")
model_elnet
# this is still causing issues with convergence without the tuneGrid specified

# let's look at coefficients for feature elimination
coef(model_elnet$finalModel, model_elnet$bestTune$lambda)
# lot of vars pushed to zero with lasso being the best model using Mean_F1 as the metric for tuning
# many of the vars pushed to zero were very important for RF model, including the various distance measures

# Take 1: SVM training ---------------------------------------------------------------------

model_svm = train(intervention ~ ., 
                    data=df.ml.train, 
                    method='svmRadial', # not sure if this is the right option from kernlab
                    tuneLength = 15,
                    maxit = 100000,
                    trControl = fitControl, 
                    metric = "Mean_F1")
model_svm

# Take 1: xgBoost DART training ------------------------------------------------------------

# defining a tuning grid
# upped eta, gamma, and colsample
tune_grid_p <- expand.grid(
  nrounds = 100, # number of boosting rounds
  max_depth = c(3, 6, 9), # max depth of trees
  eta = 0.3, # learning rate, how much each new tree contributes to final model, low to prevent overfitting
  gamma = 1, # might need to increase to make simpler trees and prevent overfitting
  subsample = 0.8,
  colsample_bytree = seq(0.5, 1, 0.1), # reducing complexity of each tree
  rate_drop = 0.1,
  skip_drop = 0.5,
  min_child_weight = 1 # might increase to prevent overfitting
)

#revising the fit control
fitControl_p <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = FALSE,
  allowParallel = TRUE,
  returnData = FALSE,
  returnResamp = "all",
  savePredictions = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

# gonna try parallelization to speed this up
# setting number of cores
num_cores <- detectCores()

# registering cores
registerDoParallel(num_cores)

model_xgbdart <- train(
  intervention ~ .,
  data = df.ml.train,
  method = 'xgbDART',
  tuneGrid = tune_grid_p,
  trControl = fitControl_p,
  metric = "Mean_F1",
  verbose = FALSE,
  nthread = 1
)

#stopCluster(num_cores)
stopImplicitCluster()

model_xgbdart

prob.xgbdart <- predict(model_xgbdart, newdata = df.ml.train, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3) # forgot why I did this..

# Take 1: Comparing models ------------------------------------------------

models_compare <- resamples(list(RF=model_rf, ELNET=model_elnet, SVM=model_svm, XGB=model_xgbdart))

summary(models_compare)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)
# the statistics here on mean precision are very misleading

# bottom line: Elnet peforms the worst. XGBoost, SVM, and RF look somewhat promising
# XGB best on precision (false positives), SVM best on recall (false negatives)

# RF model critique graphs ----------------------------------------------------------------

df.rf.graph <- bind_cols(df.ml.train, prob.rf)

# time to try model diagnostics from Colaresi and Mahmood 2017 ("Do the Robot)
# bit tricky because they only deal with binary outcome

df.rf.graph <- df.rf.graph %>% 
  mutate(int_neutral = ifelse(intervention=="neutral", 1, 0)) %>%
  mutate(int_gov = ifelse(intervention=="gov", 1, 0)) %>%
  mutate(int_reb = ifelse(intervention=="rebel", 1, 0))

# bringing in id vars
df.rf.graph <- cbind(df.rf.graph, trainID)

df.rf.graph <- df.rf.graph %>%
  mutate(target = countrycode(ccode1, "cown", "country.name")) %>%
  mutate(intervener = countrycode(ccode2, "cown", "country.name"))

# plot: gov
rfdiag_gov <- DiagPlot(
  f=df.rf.graph$gov,
  y=df.rf.graph$int_gov,
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Random Forest model of government-sided intervention."
)

pdf("_output/_figures/rf_diagnostic_gov.pdf")
grid.draw(rfdiag_gov)
dev.off()

# plot: reb
rfdiag_reb <- DiagPlot(
  f=df.rf.graph$reb,
  y=df.rf.graph$int_reb,
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Random Forest model of rebel-sided intervention."
)

pdf("_output/_figures/rf_diagnostic_reb.pdf")
grid.draw(rfdiag_reb)
dev.off()


# XGBoost DART model critique graphs --------------------------------------

df.xgbdart.graph <- bind_cols(df.ml.train, prob.xgbdart)

df.xgbdart.graph <- df.xgbdart.graph %>% 
  mutate(int_neutral = ifelse(intervention=="neutral", 1, 0)) %>%
  mutate(int_gov = ifelse(intervention=="gov", 1, 0)) %>%
  mutate(int_reb = ifelse(intervention=="rebel", 1, 0))

# bringing in id vars
df.xgbdart.graph <- cbind(df.xgbdart.graph, trainID)

df.xgbdart.graph <- df.xgbdart.graph %>%
  mutate(target = countrycode(ccode1, "cown", "country.name")) %>%
  mutate(intervener = countrycode(ccode2, "cown", "country.name"))

# plot: gov
xgbdiag_gov <- DiagPlot(
  f=df.xgbdart.graph$gov,
  y=df.xgbdart.graph$int_gov,
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="XGBoost DART model of government-sided intervention."
)

pdf("_output/_figures/xgb_diagnostic_gov.pdf")
grid.draw(xgbdiag_gov)
dev.off()

# plot: reb
xgbdiag_reb <- DiagPlot(
  f=df.xgbdart.graph$reb,
  y=df.xgbdart.graph$int_reb,
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="XGBoost DART model of rebel-sided intervention."
)

pdf("_output/_figures/xgb_diagnostic_reb.pdf")
grid.draw(xgbdiag_reb)
dev.off()


# Take 2: New variables ---------------------------------------------------

# some common conflicts in the diagnostics of false predictions: Ethiopia, Myanmar, India

# potential new variables: ccode2 already intervening in other conflict, ccode1 intervening in ccode2 (lagged), UN operation

# get new df created by pulling in id
df.ml.train2 <- cbind(df.ml.train, trainID)

# now create var for existing conflict when current conflict started
# creating df for conf-year
ucdp_confyear <- ucdpexternal %>% 
  rename(., year = ywp_year, ccode1 = locationid1) %>%
  select(conflictID, ccode1, year) %>%
  distinct()

df.ml.train2 <- df.ml.train2 %>%
  mutate(ongoingconf = mapply(function(a, c, sy) {
    any(a == ucdp_confyear$ccode1 & c!=ucdp_confyear$conflictID &
          sy == ucdp_confyear$year)
  }, ccode1, conflictID, year))
table(df.ml.train2$ongoingconf)

# creating var for whether ccode2 is currently involved in another conflict in ccode2
# start by creating df from ucdpexternal on the conf-dyad-year level
ucdp_confdyadyear <- ucdpexternal %>% filter(., external_alleged==0 & external_nameid<1000 & external_exists==1) %>%
  mutate(govrec = ifelse(str_detect(ywp_name, "^Government"), 1, 0)) %>%
  mutate(rebrec = ifelse(govrec==0, 1, 0)) %>%
  subset(., select = c("locationid1", "conflictID", "external_nameid", "govrec", "rebrec", "ywp_year")) %>%
  rename(., ccode2 = external_nameid) %>%
  rename(., ccode1 = locationid1) %>%
  rename(., year = ywp_year) %>%
  group_by(ccode1, ccode2, conflictID, year) %>%
  summarise(govsup = max(govrec), rebsup = max(rebrec)) %>%
  ungroup()

# now creating the new variable for existing/recent ccode2 gov support to ccode1
df.ml.train2 <- df.ml.train2 %>%
  mutate(otherconfgov = mapply(function(a, b, c, sy) {
    any(a == ucdp_confdyadyear$ccode1 & b == ucdp_confdyadyear$ccode2 &
          c!=ucdp_confdyadyear$conflictID &
          ucdp_confdyadyear$govsup==1 &
          sy>=ucdp_confdyadyear$year & sy <= ucdp_confdyadyear$year +3) # can't be more than three years
  }, ccode1, ccode2, conflictID, year))
table(df.ml.train2$otherconfgov)

# existing/recent rebel support
df.ml.train2 <- df.ml.train2 %>%
  mutate(otherconfreb = mapply(function(a, b, c, sy) {
    any(a == ucdp_confdyadyear$ccode1 & b == ucdp_confdyadyear$ccode2 &
          c!=ucdp_confdyadyear$conflictID &
          ucdp_confdyadyear$rebsup==1 &
          sy>=ucdp_confdyadyear$year & sy <= ucdp_confdyadyear$year +3)
  }, ccode1, ccode2, conflictID, year))
table(df.ml.train2$otherconfreb)

# let's create vars for ccode1 intervening in ccode2 in recent years
df.ml.train2 <- df.ml.train2 %>%
  mutate(targetgovint = mapply(function(a, b, sy) {
    any(a == ucdp_confdyadyear$ccode2 & b == ucdp_confdyadyear$ccode1 &
          ucdp_confdyadyear$govsup==1 &
          sy>ucdp_confdyadyear$year & sy<= ucdp_confdyadyear$year +3)
  }, ccode1, ccode2, year))
table(df.ml.train2$targetgovint)

# ccode1 intervening on reb side in ccode2 in recent years
df.ml.train2 <- df.ml.train2 %>%
  mutate(targetrebint = mapply(function(a, b, sy) {
    any(a == ucdp_confdyadyear$ccode2 & b == ucdp_confdyadyear$ccode1 &
          ucdp_confdyadyear$rebsup==1 &
          sy>ucdp_confdyadyear$year & sy<= ucdp_confdyadyear$year +3)
  }, ccode1, ccode2, year))
table(df.ml.train2$targetrebint)

# to get numeric instead of boolean
df.ml.train2 %<>% mutate_if(is.logical, as.numeric)

# dropping the ID vars again
df.ml.train2 <- subset(df.ml.train2, select = -c(ccode1, ccode2, conflictID, year))

# storing rhs and outcome for later, take 2
x = df.ml.train2[, -1]
y = df.ml.train2$intervention


# Take 2: RF train --------------------------------------------------------

set.seed(2)

fitControl <- trainControl(
  method = 'cv', #k-fold cross validation
  number = 5,
  savePredictions = 'final',
  #  classProbs = T, # I want predictions rather than probabilities
  summaryFunction = multiClassSummary
)

model_rf2 = train(intervention ~ ., data=df.ml.train2, method='rf', tuneLength = 10, trControl = fitControl, metric = "Mean_F1")
model_rf2

# pulling out predictions for model critique below
prob.rf2 <- predict(model_rf2, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3) # forgot why I did this..

# looking at variable importance
varimp_rf2 <- varImp(model_rf2)

plot(varimp_rf, main = "Variable importance with Random Forest")

# Take 2: SVM training ----------------------------------------------------

model_svm2 = train(intervention ~ ., 
                  data=df.ml.train2, 
                  method='svmRadial', # not sure if this is the right option from kernlab
                  tuneLength = 30,
                  maxit = 100000,
                  trControl = fitControl, 
                  metric = "Mean_F1")
model_svm2


# Take 2: XGBoost DART training -------------------------------------------

tune_grid_p <- expand.grid(
  nrounds = 200, # number of boosting rounds
  max_depth = c(3, 6, 9), # max depth of trees
  eta = 0.3, # learning rate, how much each new tree contributes to final model, low to prevent overfitting
  gamma = 1, # might need to increase to make simpler trees and prevent overfitting
  subsample = 0.8,
  colsample_bytree = seq(0.5, 1, 0.1), # reducing complexity of each tree
  rate_drop = 0.1,
  skip_drop = 0.5,
  min_child_weight = 1 # might increase to prevent overfitting
)

#revising the fit control
fitControl_p <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = FALSE,
  allowParallel = TRUE,
  returnData = FALSE,
  returnResamp = "all",
  savePredictions = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

# gonna try parallelization to speed this up
# setting number of cores
num_cores <- detectCores()

# registering cores
registerDoParallel(num_cores)

model_xgbdart2 <- train(
  intervention ~ .,
  data = df.ml.train2,
  method = 'xgbDART',
  tuneGrid = tune_grid_p,
  trControl = fitControl_p,
  metric = "Mean_F1",
  verbose = FALSE,
  nthread = 1
)

#stopCluster(num_cores)
stopImplicitCluster()

model_xgbdart2

# pulling out predictions for model critique graph
prob.xgbdart2 <- predict(model_xgbdart2, newdata = df.ml.train2, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3) #renaming predicted prob vars

# Take 2: Compare models --------------------------------------------------

models_compare2 <- resamples(list(RF=model_rf2, SVM=model_svm2, XGB=model_xgbdart2))

summary(models_compare2)

models_compare1_2 <- resamples(list(RF1=model_rf, RF2=model_rf2, SVM1=model_svm, SVM2=model_svm2, XGB1=model_xgbdart, XGB2=model_xgbdart2))
summary(models_compare1_2)


# Take 2: RF model critique graph --------------------------------------------------

df.rf.graph <- bind_cols(df.ml.train2, prob.rf2)

df.rf.graph <- df.rf.graph %>% 
  mutate(int_neutral = ifelse(intervention=="neutral", 1, 0)) %>%
  mutate(int_gov = ifelse(intervention=="gov", 1, 0)) %>%
  mutate(int_reb = ifelse(intervention=="rebel", 1, 0))

# bringing in id vars
df.rf.graph <- cbind(df.rf.graph, trainID)

df.rf.graph <- df.rf.graph %>%
  mutate(target = countrycode(ccode1, "cown", "country.name")) %>%
  mutate(intervener = countrycode(ccode2, "cown", "country.name"))

# plot: gov
rfdiag_gov <- DiagPlot(
  f=df.rf.graph$gov,
  y=df.rf.graph$int_gov,
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Random Forest model of government-sided intervention (take 2)."
)

pdf("_output/_figures/rf_diagnostic_gov_take2.pdf")
grid.draw(rfdiag_gov)
dev.off()

# plot: reb
rfdiag_reb <- DiagPlot(
  f=df.rf.graph$reb,
  y=df.rf.graph$int_reb,
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Random Forest model of rebel-sided intervention (take 2)."
)

pdf("_output/_figures/rf_diagnostic_reb_take2.pdf")
grid.draw(rfdiag_reb)
dev.off()

# Take 2: XGBoost DART model critique graph -------------------------------

df.xgbdart.graph <- bind_cols(df.ml.train2, prob.xgbdart2)

df.xgbdart.graph <- df.xgbdart.graph %>% 
  mutate(int_neutral = ifelse(intervention=="neutral", 1, 0)) %>%
  mutate(int_gov = ifelse(intervention=="gov", 1, 0)) %>%
  mutate(int_reb = ifelse(intervention=="rebel", 1, 0))

# bringing in id vars
df.xgbdart.graph <- cbind(df.xgbdart.graph, trainID)

df.xgbdart.graph <- df.xgbdart.graph %>%
  mutate(target = countrycode(ccode1, "cown", "country.name")) %>%
  mutate(intervener = countrycode(ccode2, "cown", "country.name"))

# plot: gov
xgbdiag_gov <- DiagPlot(
  f=df.xgbdart.graph$gov,
  y=df.xgbdart.graph$int_gov,
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="XGBoost DART model of government-sided intervention (take 2)."
)

pdf("_output/_figures/xgb_diagnostic_gov_take2.pdf")
grid.draw(xgbdiag_gov)
dev.off()

# plot: reb
xgbdiag_reb <- DiagPlot(
  f=df.xgbdart.graph$reb,
  y=df.xgbdart.graph$int_reb,
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.4,
  right_margin=9,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="XGBoost DART model of rebel-sided interventio (take 2)n."
)

pdf("_output/_figures/xgb_diagnostic_reb_take2.pdf")
grid.draw(xgbdiag_reb)
dev.off()

# observations: significant improvement on false negatives on gov AND reb support
# takeaway: got a lot better at correctly predicting intervention

# Test pre-processing -----------------------------------------------------------

# # THIS WILL NEED TO BE UPDATED FOR TAKE 2
# 
# # start by prepping test data with same pre-processing as training data
# # impute missing data
# df.ml.test2 <- predict(missing_model, df.ml.test)
# 
# # transform vars
# df.ml.test3 <- predict(range_model, df.ml.test2)
# 
# # now cutting vars that were eliminated via RFE
# 
# df.ml.test3 <- df.ml.test3[, which((names(df.ml.test3) %in% rfe.cut)==FALSE)]
# 
# # now need to add in new vars for take 2, first adding in testID vars

# RF test -----------------------------------------------------------

# # now for RF model test:
# predicted_rf <- predict(model_rf, df.ml.test3)
# head(predicted_rf)
# 
# # Confusion matrix
# confusionMatrix(reference = df.ml.test3$intervention, data = predicted_rf, mode='everything', positive='MM')
# 
# # creating plot of confusion matrix: work in progress!
# tp_rf <- data.frame(
#   obs = df.ml.test3$intervention,
#   pred = predicted_rf
# )
# 
# cm_rf <- conf_mat(tp_rf, obs, pred)
# 
# autoplot(cm_rf, type = "heatmap") +
#   scale_fill_gradient(low="#D6EAF8",high = "#2E86C1")

# SVM test --------------------------------------------------------
# predicted_svm <- predict(model_svm, df.ml.test3)
# head(predicted_svm)
# 
# confusionMatrix(reference = df.ml.test3$intervention, data = predicted_svm, mode='everything', positive='MM')

# XGBoost DART test -----------------------------------------------

# predicted_xgb <- predict(model_xgbdart, df.ml.test3)
# head(predicted_xgb)
# 
# confusionMatrix(reference = df.ml.test3$intervention, data = predicted_xgb, mode='everything', positive='MM')
