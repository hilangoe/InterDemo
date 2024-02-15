# InterDemo
# script for training various ml models of civil war intervention
# this is version two, creating a better workflow and standard procedure for preprocessing etc.

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
library(haven)
#library(rminer)
library(AICcmodavg)
library(DataExplorer)
#library(fastAdaboost) #not available anymore, but need it for Adaboost algo
library(readxl)
library(readstata13)
library(pdp)

# NOTE:
# add suite of interactions, run through lasso, cut, then RF again
# bag svm: reduced space svm, iteration ???
# need to reduce dimensionality as well as add more features

# setting theme for plots
th <- theme_light()

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
  select(., -c("ucdponset", "rivalryno", "rivalryname", "styear", "endyear", "govsupport", "rebsupport", "govsupportyear", "rebsupportyear", "bothsides", "geo_NA", "region", "type1", "type2", "type3", 'govrec', 'rebrec')) %>%
  select(intervention, everything())
# moving DV first in df

# dropping observations with intervention on both sides in the same year
df.ml <- df.ml[df.ml$intervention!=3, ]
df.ml$intervention <- factor(df.ml$intervention,
                                        levels = c(0, 1, 2))

# dropping observations with missing country codes on either side
df.ml <- df.ml %>%
  filter(., !is.na(ccode1)) %>%
  filter(., !is.na(ccode2))

# # need to fix the level names cause caret can't handle the numerical values
levels(df.ml$intervention) <- c("neutral", "gov", "rebel")


# need to fix pol_rel
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

head(df.ml.train)
class(df.ml.train$growth_wdi_1)

skimmed <- skim(df.ml.train)

skimmed[2:16,]

skimmed[17:41,]

skimmed[42:56,]

skimmed[57:71,]

skimmed[72:77,] # might not be correct final column number

# overall, pretty good coverage on most variables except the San-Akca ones

# let's look at the correlation plot first
corr_map <- DataExplorer::plot_correlation(df.ml.train)
# looks like the trade, nmc, v-dem, and v-dem distance measures are highly correlated

ggsave(file = "_output/_figures/corr_map.png", corr_map, width = 24, height = 24, dpi = 300)


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


# Benchmark ---------------------------------------------------------------

# training the libdem model with cv as a benchmark for the ml models

# defining the model formula
libdem_model <-  "intervention ~ libdemdist + v2x_libdem1 + v2x_libdem2 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop1 + upop2 + cinc1 + cinc2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear" %>%
  as.formula(.)
libdem_model

# Set up 5-fold cross-validation
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  summaryFunction = multiClassSummary
)


set.seed(3)
model_mlogit = train(
  libdem_model, 
  data=df.ml.train, 
  method='multinom', 
  trControl = fitControl, 
  metric = "Mean_F1")

print(model_mlogit)
# Mean_F1 around 0.37

# Recursive feature elimination -------------------------------------------

# starting RFE
set.seed(46)
options(warn=-1)

subsets <- c(65, 68, 71, 74, 77)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   number = 10,
                   verbose = FALSE)

# parallelization to speed this up
# setting number of cores
num_cores <- detectCores()

# registering cores
registerDoParallel(num_cores)

lmProfile <- rfe(x=df.ml.train[, 2:77], y=df.ml.train$intervention,
                 sizes = subsets,
                 rfeControl = ctrl)
stopImplicitCluster()
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

pdf("_output/_figures/rf1_varimp.pdf")
plot(varimp_rf, main = "Variable importance with Random Forest")
dev.off()

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

# looking at variable importance

varimp_xgb <- varImp(model_xgbdart)

pdf("_output/_figures/xgb1_varimp.pdf")
plot(varimp_xgb, main = "Variable importance with XGBoost DART")
dev.off()

# grabbing best tuning
best_tuning <- model_xgbdart$bestTune


# Take 1: Comparing models ------------------------------------------------

# saving models
models_take1 <- list(model_rf, model_svm, model_xgbdart)
save(models_take1, file = "_output/_models/models_take1.RData")

models_compare <- resamples(list(RF=model_rf, SVM=model_svm, XGB=model_xgbdart))

summary(models_compare)

## ggplot2 version

models_compare$values %>%
  select(1, ends_with("~Accuracy")) %>%
  gather(model, Accuracy, -1) %>%
  mutate(model = sub("~Accuracy", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Accuracy, y = model, fill = model)) -> p1

p1 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare$values %>%
  select(1, ends_with("F1")) %>%
  gather(model, F1, -1) %>%
  mutate(model = sub("~Mean_F1", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = F1, y = model, fill = model)) -> p2

p2 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare$values %>%
  select(1, ends_with("Precision")) %>%
  gather(model, Precision, -1) %>%
  mutate(model = sub("~Mean_Precision", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Precision, y = model, fill = model)) -> p3

p3 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare$values %>%
  select(1, ends_with("Recall")) %>%
  gather(model, Recall, -1) %>%
  mutate(model = sub("~Mean_Recall", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Recall, y = model, fill = model)) -> p4

p4 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

plot_list <- list(p1, p2, p3, p4)

# putting the graphs together
model_comp_boxplots1 <- patchwork::wrap_plots(plot_list, ncol = 2, guides = 'collect') &
  theme(legend.position = 'bottom') & scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9")) & th & guides(fill = FALSE)
ggsave(file = "_output/_figures/model_comp_boxplots1.png", model_comp_boxplots1, width = 8, height = 6, dpi = 300)

# ## lattice version
# 
# 
# facet_labels <- c('Acc.', 'Kappa', 'Bal. acc.',
#                   'Det. rate', 'F1', 'Neg. pred. val',
#                   'Pos. pred. val.', 'Prec.', 'Recall',
#                   'Sens.', 'Specif.')
# 
# scales <- list(x=list(relation="free"), y=list(relation="free"))
# 
# take1_comp <- bwplot(models_compare, scales=scales)
# the statistics here on mean precision are very misleading

# bottom line: XGBoost, SVM, and RF look somewhat promising
# XGB best on precision (false positives), SVM best on recall (false negatives)

png("_output/_figures/take1_comp.png")
plot(take1_comp, main = "Performance of model with different algorithms")
dev.off()

# confusion matrices

confusion_rf1 <- model_rf[[5]]

confusion_xgb1 <- model_xgbdart[[5]]
# need to filter here, because I did savePredictions==TRUE on tune
confusion_xgb1 <- confusion_xgb1 %>%
  filter(nrounds == best_tuning$nrounds &
           max_depth == best_tuning$max_depth &
           eta == best_tuning$eta &
           gamma == best_tuning$gamma &
           subsample == best_tuning$subsample &
           colsample_bytree == best_tuning$colsample_bytree &
           rate_drop == best_tuning$rate_drop &
           skip_drop == best_tuning$skip_drop &
           min_child_weight == best_tuning$min_child_weight)

# let's look and compare
conf_rf1 <- table(confusion_rf1$obs, confusion_rf1$pred)
conf_rf1

conf_xgb1 <- table(confusion_xgb1$obs, confusion_xgb1$pred)
conf_xgb1


# Feature elimination: Elastic net training ---------------------------------------------------

## elnet (glmnet)

# setting number of cores
num_cores <- detectCores()

# registering cores
registerDoParallel(num_cores)

model_elnet = train(intervention ~ ., 
                    data=df.ml.train, 
                    method='glmnet', 
                    family='multinomial',
                    type.multinomial='grouped',
                    #                    tuneLength = 5,
                    tuneGrid = expand.grid(
                      alpha = seq(0, 1, by = 0.1), 
                      lambda = seq(0, 1, by = 0.005)), #made penalty more fine-grained
                    maxit = 100000,
                    trControl = fitControl, metric = "Mean_F1")
# this is still causing issues with convergence without the tuneGrid specified
# running into memory issues in R with larger tune grid
stopImplicitCluster()

model_elnet

# taking a closer look at results of tune
df_elnet_results <- model_elnet[[4]]
filter(df_elnet_results, lambda==0)
# seems like minor, random fluctuations in mean F1
filter(df_elnet_results, lambda==0.01)
# need to graph lambda over mean_f1

# checking functional form
# setting maximum degree for polynomial regression
max_degree <- 4
# Iterate through polynomial degrees
for (degree in 1:max_degree) {
  formula <- as.formula(paste("Mean_F1 ~ poly(lambda, ", degree, ")", sep = ""))
  model_name <- paste("el", degree, sep = "")  # creating variable name
  assign(model_name, lm(data = df_elnet_results, formula = formula)) # assigning model to model name
}

# creating list of models
models <- list(el1, el2, el3, el4)
model.names <- c('el1', 'el2', 'el3', 'el4') # model names for table purposes

aictab(cand.set = models, modnames = model.names)
# definitely think polynomial is overfitting the line, so gonna go with linear (suspect it's actually some form of logarithmic function)

# Create a new dataframe for prediction with missing values
prediction_data <- data.frame(lambda = df_elnet_results$lambda)

# Calculate complete cases
complete_cases <- complete.cases(prediction_data)

# Filter out incomplete cases from prediction_data
prediction_data <- subset(prediction_data, complete_cases)

# Calculate the fitted values for each model
df_elnet_results$fit_el1 <- predict(el1, newdata = prediction_data)
df_elnet_results$fit_el2 <- predict(el2, newdata = prediction_data)
df_elnet_results$fit_el3 <- predict(el3, newdata = prediction_data)
df_elnet_results$fit_el4 <- predict(el4, newdata = prediction_data)

# Create the ggplot scatter plot
p <- ggplot(df_elnet_results, aes(x = lambda, y = Mean_F1)) +
  geom_point() +
  xlim(0, 0.08) +
  ylim(0.34, 0.41) +
  theme_classic()

# Add custom fitted lines to the ggplot
elnet_lambda <- p + 
  geom_line(data = df_elnet_results, aes(x = lambda, y = fit_el1, color = "1")) +
  geom_line(data = df_elnet_results, aes(x = lambda, y = fit_el2, color = "2")) +
  geom_line(data = df_elnet_results, aes(x = lambda, y = fit_el3, color = "3")) +
  geom_line(data = df_elnet_results, aes(x = lambda, y = fit_el4, color = "4")) +
  scale_color_manual(values = c("purple", "#A1DAB4", "#41B6C4", "magenta")) +
  labs(color = "Polynomial")  # Add a label to the legend

png("_output/_figures/elnet_lambda.png")
plot(elnet_lambda, main = "Mean F1 score and regularization penalty (lambda)")
dev.off()


elnet_alpha <- ggplot(df_elnet_results, aes(x = alpha, y = Mean_F1)) +
  geom_point() +
  geom_smooth(method = "lm", formula = y ~ x, color = "darkblue", fill = "lightblue", fullrange = TRUE) +
  theme_classic()

png("_output/_figures/elnet_alpha.png")
plot(elnet_alpha, main = "Mean F1 score and Ridge/LASSO weight (alpha)")
dev.off()

# let's look at coefficients for feature elimination
elnet_coef <- coef(model_elnet$finalModel, model_elnet$bestTune$lambda)
elnet_coef

# doesn't work because of zero coefficients
# df_elnet_coef_tuned <- data.frame(var = elnet_coef[[1]]@Dimnames[1], 
#                                   neutral = elnet_coef[[1]]@x,
#                                   gov = elnet_coef[[2]]@x,
#                                   reb = elnet_coef[[3]]@x)

elnet_neutral <- as.data.frame(as(elnet_coef[[1]], "matrix"))
elnet_gov <- as.data.frame(as(elnet_coef[[2]], "matrix"))
elnet_reb <- as.data.frame(as(elnet_coef[[3]], "matrix"))

df_elnet_coef_tuned <- cbind(elnet_neutral, elnet_gov, elnet_reb) %>%
  rename(neutral = 1, gov = 2, reb = 3)

print(df_elnet_coef_tuned)
df_elnet_coef_tuned[rowSums(df_elnet_coef_tuned == 0) == ncol(df_elnet_coef_tuned), ]
zero_coeff_vars <- names(df_elnet_coef_tuned)[colSums(df_elnet_coef_tuned == 0) == nrow(df_elnet_coef_tuned)]

elnet_cut <- rownames(df_elnet_coef_tuned)[rowSums(df_elnet_coef_tuned == 0) == ncol(df_elnet_coef_tuned)]

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
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,df.rf.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.5,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting government support (RF)"
)

ggsave(file = "_output/_figures/rf1_diagplot_gov.png", rfdiag_gov, width = 6, height = 6, dpi = 300)

# plot: reb
rfdiag_reb <- DiagPlot(
  f=df.rf.graph$reb,
  y=df.rf.graph$int_reb,
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,df.rf.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.5,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting rebel support (RF)"
)

ggsave(file = "_output/_figures/rf1_diagplot_reb.png", rfdiag_reb, width = 6, height = 6, dpi = 300)


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
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener,df.xgbdart.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.5,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting government support (XGB)"
)

ggsave(file = "_output/_figures/xgb1_diagplot_gov.png", xgbdiag_gov, width = 6, height = 6, dpi = 300)

# plot: reb
xgbdiag_reb <- DiagPlot(
  f=df.xgbdart.graph$reb,
  y=df.xgbdart.graph$int_reb,
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener,df.xgbdart.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.55,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting rebel support (XGB)"
)

ggsave(file = "_output/_figures/xgb1_diagplot_reb.png", xgbdiag_reb, width = 6, height = 6, dpi = 300)

# Take 2: Feature selection/elimination ---------------------------------------------------

# cutting dropped vars from elastic net model (elnet_cut)
df.ml.train2 <- df.ml.train %>%
  select(-c(elnet_cut))

# from diagplot: some common conflicts in the diagnostics of false predictions: Ethiopia, Myanmar, India

## Creating new variables
# get new df created by pulling in id
df.ml.train2 <- cbind(df.ml.train2, trainID)

## NEW VAR: ongoingconf (is there an existing conflict in ccode1?)
# now create var for existing conflict when current conflict started
# creating df for conf-year
ucdp_confyear <- ucdpexternal %>% 
  select(conflictID, ccode1, year) %>%
  distinct()

df.ml.train2 <- df.ml.train2 %>%
  mutate(ongoingconf = mapply(function(a, c, sy) {
    any(a == ucdp_confyear$ccode1 & c!=ucdp_confyear$conflictID &
          sy == ucdp_confyear$year)
  }, ccode1, conflictID, year))
table(df.ml.train2$ongoingconf)

## NEW VAR: otherconfgov (is ccode2 now or recently supporting ccode1 gov?)
# creating var for whether ccode2 is currently involved in another conflict in ccode2
# start by creating df from ucdpexternal on the conf-dyad-year level
ucdp_confdyadyear <- ucdpexternal %>% filter(., ext_alleged==0 & ext_nonstate==0 & ext_sup==1 & ext_elements==0) %>%
  mutate(govrec = ifelse(actor_nonstate==0, 1, 0)) %>%
  mutate(rebrec = ifelse(actor_nonstate==1, 1, 0)) %>%
  left_join(., actor_translate, by = c('ext_id' = 'new_id')) %>%
  rename(ccode2 = old_id) %>%
  subset(., select = c("ccode1", "conflictID", "ccode2", "govrec", "rebrec", "year")) %>%
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

## NEW VAR: otherconfreb (is ccode2 now or recently supporting ccode1 reb?)
# existing/recent rebel support
df.ml.train2 <- df.ml.train2 %>%
  mutate(otherconfreb = mapply(function(a, b, c, sy) {
    any(a == ucdp_confdyadyear$ccode1 & b == ucdp_confdyadyear$ccode2 &
          c!=ucdp_confdyadyear$conflictID &
          ucdp_confdyadyear$rebsup==1 &
          sy>=ucdp_confdyadyear$year & sy <= ucdp_confdyadyear$year +3)
  }, ccode1, ccode2, conflictID, year))
table(df.ml.train2$otherconfreb)

## NEW VAR: targetgovint (has ccode1 intervened recently in ccode2 on gov side?)
# let's create vars for ccode1 intervening in ccode2 in recent years
df.ml.train2 <- df.ml.train2 %>%
  mutate(targetgovint = mapply(function(a, b, sy) {
    any(a == ucdp_confdyadyear$ccode2 & b == ucdp_confdyadyear$ccode1 &
          ucdp_confdyadyear$govsup==1 &
          sy>ucdp_confdyadyear$year & sy<= ucdp_confdyadyear$year +3)
  }, ccode1, ccode2, year))
table(df.ml.train2$targetgovint)

## NEW VAR: targetrebint (has ccode1 intervened recently in ccode2 on reb side?)
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

anyNA(df.ml.train2)
# one obs with NAs
# pre-processing again to fix it

# imputing missing data
missing_model <- preProcess(as.data.frame(df.ml.train2), method='knnImpute', k = 3)

df.ml.train2 <- predict(missing_model, df.ml.train2)
anyNA(df.ml.train2)

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


model_rf2 = train(intervention ~ ., 
                  data=df.ml.train2, 
                  method='rf', 
                  tuneLength = 20, 
                  trControl = fitControl, 
                  metric = "Mean_F1")
model_rf2

# pulling out predictions for model critique below
prob.rf2 <- predict(model_rf2, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3) # forgot why I did this..

# looking at variable importance
varimp_rf2 <- varImp(model_rf2)

pdf("_output/_figures/rf2_varimp.pdf")
plot(varimp_rf2, main = "Variable importance with Random Forest (take 2)")
dev.off()

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
  nrounds = c(100, 150, 200, 250), # number of boosting rounds, tuning
  max_depth = c(3, 6, 9), # max depth of trees
  eta = 0.3, # default learning rate, how much each new tree contributes to final model, low to prevent overfitting
  gamma = 1, # leaving low because elnet suggests no regularization; might need to increase to make simpler trees and prevent overfitting
  subsample = 0.8,
  colsample_bytree = seq(0.5, 1, 0.1), # reducing complexity of each tree
  rate_drop = 0.1,
  skip_drop = 0.5,
  min_child_weight = c(0, 5, 10) # might increase to prevent overfitting
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

# grabbing the best tuning parameters
best_tuning <- model_xgbdart2$bestTune
best_tuning

# grabbing the performance metrics for each tuning iteration
tuning_results <- model_xgbdart2$results

# Filter the tuning_results for the best tune
best_tuning_results <- tuning_results[tuning_results$nrounds == best_tuning$nrounds &
                                        tuning_results$max_depth == best_tuning$max_depth &
                                        tuning_results$eta == best_tuning$eta &
                                        tuning_results$gamma == best_tuning$gamma &
                                        tuning_results$subsample == best_tuning$subsample &
                                        tuning_results$colsample_bytree == best_tuning$colsample_bytree &
                                        tuning_results$rate_drop == best_tuning$rate_drop &
                                        tuning_results$skip_drop == best_tuning$skip_drop &
                                        tuning_results$min_child_weight == best_tuning$min_child_weight, ]

# View the results for the best tune
print(best_tuning_results)

# pulling out predictions for model critique graph
prob.xgbdart2 <- predict(model_xgbdart2, newdata = df.ml.train2, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3) #renaming predicted prob vars

varimp_xgb2 <- varImp(model_xgbdart2)

pdf("_output/_figures/xgb2_varimp.pdf")
plot(varimp_xgb2, main = "Variable importance with XGBoost DART (take 2)")
dev.off()

# Take 2: Compare models --------------------------------------------------

# saving models
models_take2 <- list(model_rf2, model_svm2, model_xgbdart2)
save(models_take2, file = "_output/_models/models_take2.RData")

models_compare2 <- resamples(list(RF=model_rf2, SVM=model_svm2, XGB=model_xgbdart2))

summary(models_compare2)

models_compare1_2 <- resamples(list(RF1=model_rf, RF2=model_rf2, SVM1=model_svm, SVM2=model_svm2, XGB1=model_xgbdart, XGB2=model_xgbdart2))
summary(models_compare1_2)

# boxplots

models_compare2$values %>%
  select(1, ends_with("~Accuracy")) %>%
  gather(model, Accuracy, -1) %>%
  mutate(model = sub("~Accuracy", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Accuracy, y = model,fill = model)) -> p1

p1 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare2$values %>%
  select(1, ends_with("F1")) %>%
  gather(model, F1, -1) %>%
  mutate(model = sub("~Mean_F1", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = F1, y = model,fill = model)) -> p2

p2 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare2$values %>%
  select(1, ends_with("Precision")) %>%
  gather(model, Precision, -1) %>%
  mutate(model = sub("~Mean_Precision", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Precision, y = model,fill = model)) -> p3

p3 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare2$values %>%
  select(1, ends_with("Recall")) %>%
  gather(model, Recall, -1) %>%
  mutate(model = sub("~Mean_Recall", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Recall, y = model,fill = model)) -> p4

p4 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

plot_list <- list(p1, p2, p3, p4)

# putting the graphs together
model_comp_boxplots2 <- patchwork::wrap_plots(plot_list, ncol = 2, guides = 'collect') &
  theme(legend.position = 'bottom') & scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9")) & th & guides(fill = FALSE)
ggsave(file = "_output/_figures/model_comp_boxplots2.png", model_comp_boxplots2, width = 8, height = 6, dpi = 300)

# confusion matrices

confusion_rf2 <- model_rf2[[5]]

confusion_xgb2 <- model_xgbdart2[[5]]
# need to filter here, because I did savePredictions==TRUE on tune
confusion_xgb2 <- confusion_xgb2 %>%
  filter(nrounds == best_tuning$nrounds &
           max_depth == best_tuning$max_depth &
           eta == best_tuning$eta &
           gamma == best_tuning$gamma &
           subsample == best_tuning$subsample &
           colsample_bytree == best_tuning$colsample_bytree &
           rate_drop == best_tuning$rate_drop &
           skip_drop == best_tuning$skip_drop &
           min_child_weight == best_tuning$min_child_weight)

# let's look and compare
conf_rf2 <- table(confusion_rf2$obs, confusion_rf2$pred)
conf_rf2

conf_xgb2 <- table(confusion_xgb2$obs, confusion_xgb2$pred)
conf_xgb2

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
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,df.rf.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.55,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting government support (RF, take 2)"
)

ggsave(file = "_output/_figures/rf2_diagplot_gov.png", rfdiag_gov, width = 6, height = 6, dpi = 300)

# plot: reb
rfdiag_reb <- DiagPlot(
  f=df.rf.graph$reb,
  y=df.rf.graph$int_reb,
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,df.rf.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.5,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting rebel support (RF, take 2)"
)

ggsave(file = "_output/_figures/rf2_diagplot_reb.png", rfdiag_reb, width = 6, height = 6, dpi = 300)

# takeaways: poor on Ukraine gov sided; better on rebel sided; clearer improvement overall on reb prediction
# takeaways: bad on gov-sided false negatives, better (and improving) on reb-sided false negatives
# remaining issues: false positives on gov-sided in Asia; false negatives on gov-sided in Ukraine
# remaining issues: false positives on reb-sided in Myanmar, Ethiopia; false negatives on reb-sided in Myanmar, Ethiopia, Philippines

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
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener, df.xgbdart.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.55,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting government support, (XGB, take 2)"
)

ggsave(file = "_output/_figures/xgb2_diagplot_gov.png", xgbdiag_gov, width = 6, height = 6, dpi = 300)


# plot: reb
xgbdiag_reb <- DiagPlot(
  f=df.xgbdart.graph$reb,
  y=df.xgbdart.graph$int_reb,
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener, df.xgbdart.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.55,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting rebel support, (XGB, take 2)"
)

ggsave(file = "_output/_figures/xgb2_diagplot_reb.png", xgbdiag_reb, width = 6, height = 6, dpi = 300)

# takeaways: worse on all counts it seems
# remaining issues: false positives on gov-sided in Ethiopia and Mali; false negatives on gov-sided in Ethiopia
# remaining issues: false positives on reb-sided in Ethiopia and Myanmar; false negatives on reb-sided in Ethiopia and India

# Take 2: Analyzing prediction errors -------------------------------------

# gonna create a dataframe to take a closer look at the worst predictions
rf2_worst_gov <- df.rf.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_gov-gov) %>% 
  arrange(gov) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_gov) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

rf2_worst_gov <- rf2_worst_gov %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

rf2_worst_reb <- df.rf.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_reb-reb) %>% 
  arrange(reb) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_reb) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

rf2_worst_reb <- rf2_worst_reb %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

# updated take: US involvement in nearly all big miss conflicts
# updated take: ethiopia is tricky because the conflict shifted over time, which my model can't account for

# now look at xgbdart
# gonna create a dataframe to take a closer look at the worst predictions
xgbdart2_worst_gov <- df.xgbdart.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_gov-gov) %>% 
  arrange(gov) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_gov) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

xgbdart2_worst_gov <- xgbdart2_worst_gov %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

xgbdart2_worst_reb <- df.xgbdart.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_reb-reb) %>% 
  arrange(reb) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_reb) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

xgbdart2_worst_reb <- xgbdart2_worst_reb %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

# updated take: the biggest misses on rebel are much worse than rf2 model
# updated take: ukraine and ethiopia still frequent, but more diverse set of conflicts
# updated take: lot more confusion than rf2, predicting the wrong side of the intervention

## GRAPHING

# let's look at error over time
rf2_graph_worst <- df.rf.graph %>%
  select(intervention, year, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(error_gov=abs(int_gov-gov)) %>%
  mutate(error_reb=abs(int_reb-reb)) 

ggplot(rf2_graph_worst, aes(x = year, y = error_gov)) +
  geom_point()

ggplot(rf2_graph_worst, aes(x = year, y = error_reb)) +
  geom_point()

xgbdart2_graph_worst <- df.xgbdart.graph %>%
  select(intervention, year, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(error_gov=abs(int_gov-gov)) %>%
  mutate(error_reb=abs(int_reb-reb)) 

ggplot(xgbdart2_graph_worst, aes(x = year, y = error_gov)) +
  geom_point()

ggplot(xgbdart2_graph_worst, aes(x = year, y = error_reb)) +
  geom_point()

# takeaway: error_reb seems negatively correlated with time, might be function of decolonization, third wave

# let's look at density of errors by outcome
# gov error first
ggplot(rf2_graph_worst, aes(error_gov)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)
# then reb error
ggplot(rf2_graph_worst, aes(error_reb)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)

# let's look at density of errors by outcome
# gov error first
ggplot(xgbdart2_graph_worst, aes(error_gov)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)
# then reb error
ggplot(xgbdart2_graph_worst, aes(error_reb)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)

# takeaway: rf2 seems worse..

# Take 3: New variables ---------------------------------------------------

## to add:
# age of state
# EPR: "egipgrps", "exclgrps", "exclpop", "rexclpop"
# EPR-TEK: got list, so just need to pull via mapply code
# unga ideal points estimate (proximity to US) for target and intervener
# maybe: secessionist conflicts: Ethhnic Armed Conflict, v3 (TBD)

# get new df created by pulling in id
df.ml.train3 <- cbind(df.ml.train2, trainID)

## NEW VAR: log_state_age (how old is the civil war country, per COW state membership list?)
## NEW VAR: new_state (state entered system in the last three years)

# pulling in COW state membership list
cowstates <- read.csv("_data/_raw/states2016.csv")

# drop any observations that ended before 1975
cowstates <- cowstates %>%
  filter(endyear>=1975)

duplicated(cowstates$ccode)
# no states that have exited and entered the system post-1974

cowstates <- cowstates %>%
  select(ccode, styear, endyear) %>%
  rename(ccode1 = ccode, cow_startyear = styear, cow_endyear = endyear)

df.ml.train3 <- df.ml.train3 %>%
  left_join(., cowstates, by = "ccode1")

# some validation
filter(df.ml.train3, is.na(cow_startyear))
filter(df.ml.train3, year>cow_endyear & year<2016)

df.ml.train3 <- df.ml.train3 %>%
  mutate(state_age = year - cow_startyear) %>%
  select(-c('cow_startyear', 'cow_endyear'))
summary(df.ml.train3$state_age)
ggplot(df.ml.train3, aes(state_age)) +
  geom_histogram()
# should log-transform this variable, but also include a dummy for new state (3 years or younger)

df.ml.train3 <- df.ml.train3 %>%
  mutate(log_state_age = log(state_age)) %>%
  mutate(new_state = ifelse(state_age<=3, 1, 0)) %>%
  select(., -c('state_age'))

ggplot(df.ml.train3, aes(log_state_age)) +
  geom_histogram()

## EPR vars
## NEW VAR: egip_groups_count (EGIP groups count)
## NEW VAR: excl_groups_count (MEG groups count)
## NEW VAR: exclpop (portion of total pop that's excluded)

epr_country_year <- read.csv("_data/_raw/epr_country_year.csv")

epr_country_year <- epr_country_year %>%
  mutate(ccode1 = countrycode(countries_gwid, "gwn", "cown")) 
# ambiguous match with 340 (Serbia), 711, and 816 (Vietnam)

# let's figure out these mismatches
filter(epr_country_year, countryname=="Serbia")
filter(epr_country_year, countries_gwid==340)
filter(df.ml.train3, ccode1==345)
filter(cowstates, ccode1==345)
# no separate observations for Serbia, it took over Yugoslavia's COW code

# fixing serbia
epr_country_year$ccode1[epr_country_year$countries_gwid==340] <- 345

# tibet (irrelevant)
filter(epr_country_year, countries_gwid==711)

# vietnam
filter(epr_country_year, countries_gwid==816)
filter(cowstates, ccode1==816)
filter(cowstates, ccode1==817)

# fixing vietnam
epr_country_year$ccode1[epr_country_year$countries_gwid==816] <- 816

epr_country_year <- epr_country_year %>%
  filter(year>=1975) %>%
  select(., ccode1, year, egip_groups_count, excl_groups_count, exclpop) %>%
  filter(!is.na(ccode1)) %>%
  distinct(ccode1, year, .keep_all = TRUE)

# anyNA(epr_country_year)
# filter(epr_country_year, is.na(ccode1))

df.ml.train3 <- df.ml.train3 %>%
  left_join(., epr_country_year, by = c("ccode1", "year"))

## NEW VAR: shares_ethnic_group (do ccode1 and ccode2 share an ethnic group, per EPR-TEK?)

tek <- read.csv("_data/_raw/TEK-2021.csv")

tek <- tek %>%
  mutate(ccode = countrycode(gwid, "gwn", "cown")) 

# fixing serbia
tek$ccode[tek$ccode==340] <- 345

# fixing vietnam
tek$ccode[tek$ccode==816] <- 816

tek <- tek %>%
  select(., tekid, ccode)

# constructing variable for matching ethnic groups

matching_groups <- tek %>%
  group_by(tekid) %>%
  filter(n() >= 2) %>%
  ungroup()

tek_dyads <- trainID %>%
  distinct(ccode1, ccode2, .keep_all = TRUE) %>%
  select(-c('conflictID', 'year')) %>%
  left_join(matching_groups, by = c('ccode1' = 'ccode')) %>%
  rename(tekid1 = tekid) %>%
  filter(!is.na(tekid1)) %>% # dropping obs with no tekid because those aren't relevant
  left_join(matching_groups, by = c('ccode2' = 'ccode')) %>%
  rename(tekid2 = tekid) %>%
  filter(!is.na(tekid2)) %>%
  mutate(shares_ethnic_group = ifelse(tekid1 == tekid2, 1, 0)) %>%
  select(ccode1, ccode2, shares_ethnic_group) %>%
  distinct(ccode1, ccode2, .keep_all = TRUE)
table(tek_dyads$shares_ethnic_group)
filter(tek_dyads, shares_ethnic_group==1)

anyNA(tek_dyads)

# joining

df.ml.train3 <- df.ml.train3 %>%
  left_join(., tek_dyads, by = c("ccode1", "ccode2"))

table(df.ml.train3$shares_ethnic_group)

## NEW VAR: goalindep (is a group in the conflict seeking independence?) (from FORGE)
## NEW VAR: goalauto (is a group in the conflict seeking autonomy?) (from FORGE)
## NEW VAR: goalrights (is a group in the conflict seeking rights?) (from FORGE)
## NEW VAR: goalrep (is a group in the conflict seeking reputation?) (from FORGE)
## NEW VAR: goalchange (is a group in the conflict seeking leadership change?) (from FORGE)
## NEW VAR: goaldem (is a group in the conflict seeking democracy?) (from FORGE)
## NEW VAR: ideolcom (does a group in the conflict have communist ideology?) (from FORGE)
## NEW VAR: ideolleft (does a group in the conflict have leftist ideology?) (from FORGE)
## NEW VAR: ideolright (does a group in the conflict have right-wing ideology?) (from FORGE)
## NEW VAR: ideolnat (does a group in the conflict have nativist ideology?) (from FORGE)
## NEW VAR: ideolanti (does a group in the conflict have anti ideology?) (from FORGE)
## NEW VAR: ideolrel (does a group in the conflict have religious ideology?) (from FORGE)
## NEW VAR: ethnic (is a group in the conflict supporting an ethnicity?) (from FORGE)

forge <- read.dta13("_data/_raw/forge_v1.0_public.dta")

# first I need to weed out groups that either started fighting much earlier than UCDP starts the conflict or much later
# groups that started 55 years ago are less relevant than groups that started the year of onset
forge <- forge %>%
  left_join(., trainID %>% # should probably use the whole dataframe so as to avoid having to repeat for test set
              select(conflictID, year) %>%
              distinct(), by = c('conflict_id' = "conflictID")) %>%
  mutate(discrep = abs(fightyear - year)) %>% # time discrepancy between fight year and UCDP start year
  group_by(conflict_id) %>%
  arrange(discrep) %>% # within each conflict, sorting in ascending order the time discrepancy
  mutate(closest=ifelse(row_number()<=3, 1, 0)) %>% # dummy for the three closest groups
  ungroup() %>%
  arrange(conflict_id) %>%
  select(conflict_id, closest, discrep, foundyear, fightyear, year, everything()) %>%
  filter(closest==1) %>% # this is going to filter out everyone except the three groups closest to UCDP start year
  filter(., fightyear<year+3) # filter out groups that started fighting three or more years after UCDP onset year
# seems conflictID 387 is assigned to both Angola in the 1990s and CAR (482) in 2012 in UCDP
# gonna have to manually correct that

forge_vars <- c('goalindep', 'goalauto', 'goalrights', 'goalrep', 'goalchange', 'goaldem', 'ideolcom', 'ideolleft', 'ideolright', 'ideolnat', 'ideolanti', 'ideolrel', 'ethnic')

# time to create variables, the maximum per conflict
forge_conf <- forge %>%
  rename(conflictID = conflict_id) %>%
  select(conflictID, all_of(forge_vars)) %>%
  group_by(conflictID) %>%
  summarise_all(., max, na.rm = TRUE)

# joining
df.ml.train3 <- df.ml.train3 %>%
  left_join(., forge_conf, by = c("conflictID"))
summary(df.ml.train3$goalindep)
# significant amount of missing data
skim(df.ml.train3)

# time to fix erronous data on Central African Republic
df.ml.train3$conflictID[df.ml.train3$ccode1==482]
df.ml.train3$year[df.ml.train3$ccode1==482]

# setting all of the forge vars to missing for CAR in conflictID 387
df.ml.train3 <- df.ml.train3 %>%
  mutate(across(all_of(forge_vars), ~ ifelse(ccode1==482 & conflictID==387, NA, .)))
skim(df.ml.train3)

## NEW VAR: unga_idealpoint (what is cccode1's unga-based ideal point?)
## NEW VAR: unga_us_absdiff (what is the absolute difference between ccode1's and US' idealpoints?)

unga <- read.csv("_data/_raw/IdealpointestimatesAll_Jul2023.csv")
head(unga)

# select and stuff (year = session + 1945)
unga <- unga %>%
  mutate(year = session + 1945) %>%
  rename(ccode1 = ccode, unga_idealpoint = IdealPointAll) %>%
  select(., year, ccode1, unga_idealpoint)
unga %>% group_by(ccode1, year) %>%
  mutate(dupe = n()>1) %>%
  filter(dupe==TRUE) %>%
  ungroup()
  
unga_us <- unga %>%
  filter(ccode1==2) %>%
  rename(unga_idealpoint_us = unga_idealpoint) %>%
  select(-c('ccode1'))

df.ml.train3 <- df.ml.train3 %>%
  left_join(., unga, by = c('ccode1', 'year')) %>%
  left_join(., unga_us, by = c('year')) %>%
  mutate(unga_us_absdiff = abs(unga_idealpoint-unga_idealpoint_us)) %>%
  select(-c('unga_idealpoint_us'))


# Take 3: Preprocessing ---------------------------------------------------

## preprocessing
# drop ID vars
df.ml.train3 <- subset(df.ml.train3, select = -c(ccode1, ccode2, conflictID, year))

anyNA(df.ml.train3)

# impute missing data
missing_model <- preProcess(as.data.frame(df.ml.train3), method='knnImpute', k = 3)
missing_model

df.ml.train3 <- predict(missing_model, df.ml.train3)
anyNA(df.ml.train3)

# transform to range, since some vars vary a lot in value range

range_model <- preProcess(as.data.frame(df.ml.train3), method='range')

df.ml.train3 <- predict(range_model, df.ml.train3)

skim(df.ml.train3)

# Take 3: RF train --------------------------------------------------------

set.seed(53)

fitControl <- trainControl(
  method = 'cv', #k-fold cross validation
  number = 5,
  savePredictions = 'final',
  #  classProbs = T, # I want predictions rather than probabilities
  summaryFunction = multiClassSummary
)

num_cores <- detectCores()

# registering cores
registerDoParallel(num_cores)

model_rf3 = train(intervention ~ ., 
                  data=df.ml.train3, 
                  method='rf', 
                  tuneLength = 20, 
                  trControl = fitControl, 
                  metric = "Mean_F1")
stopImplicitCluster()

model_rf3

# pulling out predictions for diagplot below
prob.rf3 <- predict(model_rf3, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3)

# creating manual variable importance plot
importance <- varImp(model_rf3, scale = FALSE)

# recasting to dataframe for manipulation, first pulling out importance value and var names
importance_values <- importance[[1]]
variables <- rownames(importance_values)

# Create a data frame
importance_df <- data.frame(Variables = variables, Importance = importance_values)
rownames(importance_df) <- NULL

importance_df <- importance_df %>%
  mutate(type = ifelse(grepl("1$", Variables), 1,
                       ifelse(grepl("2$", Variables), 2, 0)))
importance_df$type[importance_df$Variables=="geo_2"] <- 1

# setting levels
importance_df$type <- factor(importance_df$type,
                                     levels = c(0, 1, 2), 
                                     labels = c("Dyadic", "Civil war country", "Potential intervener"))

head(importance_df)

p <- ggplot(data = importance_df, aes(y = Overall, x = reorder(Variables, Overall), fill = type)) +
  geom_bar(stat = "identity", alpha = 1) +
  coord_flip() +
  th +
  labs(title = "Variable importance (Random Forest)", 
       y = "Importance score",
       x = "Feature",
       fill = "Type") +
  theme(legend.position = "bottom")
ggsave(file = "_output/_figures/rf3_varimp.png", p, width = 15, height = 12, dpi = 300)

# Take 3: SVM train -------------------------------------------------------

set.seed(33)

num_cores <- detectCores()

# registering cores
registerDoParallel(num_cores)

model_svm3 = train(intervention ~ ., 
                   data=df.ml.train3, 
                   method='svmRadial',
                   tuneLength = 30,
                   maxit = 100000,
                   trControl = fitControl, 
                   metric = "Mean_F1")
stopImplicitCluster()

model_svm3

# Take 3: XGBoost DART train ----------------------------------------------

tune_grid_p <- expand.grid(
  nrounds = c(100, 150, 200, 250), # number of boosting rounds, tuning
  max_depth = c(3, 6, 9), # max depth of trees
  eta = 0.3, # default learning rate, how much each new tree contributes to final model, low to prevent overfitting
  gamma = 1, # leaving low because elnet suggests no regularization; might need to increase to make simpler trees and prevent overfitting
  subsample = 0.8,
  colsample_bytree = seq(0.5, 1, 0.1), # reducing complexity of each tree
  rate_drop = 0.1,
  skip_drop = 0.5,
  min_child_weight = c(0, 5, 10) # might increase to prevent overfitting
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

model_xgbdart3 <- train(
  intervention ~ .,
  data = df.ml.train3,
  method = 'xgbDART',
  tuneGrid = tune_grid_p,
  trControl = fitControl_p,
  metric = "Mean_F1",
  verbose = FALSE,
  nthread = 1
)

#stopCluster(num_cores)
stopImplicitCluster()

model_xgbdart3

# grabbing the best tuning parameters
best_tuning <- model_xgbdart3$bestTune
best_tuning

# grabbing the performance metrics for each tuning iteration
tuning_results <- model_xgbdart3$results

# Filter the tuning_results for the best tune
best_tuning_results <- tuning_results[tuning_results$nrounds == best_tuning$nrounds &
                                        tuning_results$max_depth == best_tuning$max_depth &
                                        tuning_results$eta == best_tuning$eta &
                                        tuning_results$gamma == best_tuning$gamma &
                                        tuning_results$subsample == best_tuning$subsample &
                                        tuning_results$colsample_bytree == best_tuning$colsample_bytree &
                                        tuning_results$rate_drop == best_tuning$rate_drop &
                                        tuning_results$skip_drop == best_tuning$skip_drop &
                                        tuning_results$min_child_weight == best_tuning$min_child_weight, ]

# View the results for the best tune
print(best_tuning_results)

# pulling out predictions for model critique graph
prob.xgbdart3 <- predict(model_xgbdart3, newdata = df.ml.train3, type = "prob") %>%
  rename(neutral = 1, gov = 2, reb = 3) #renaming predicted prob vars

# creating manual variable importance plot
importance <- xgb.importance(model = model_xgbdart3$finalModel) 
importance <- importance[order(-Gain),]

head(importance)

importance_df <- importance %>%
  mutate(type = ifelse(grepl("1$", Feature), 1,
                       ifelse(grepl("2$", Feature), 2, 0)))
importance_df$type[importance_df$Feature=="geo_2"] <- 1

# setting levels
importance_df$type <- factor(importance_df$type,
                             levels = c(0, 1, 2), 
                             labels = c("Dyadic", "Civil war country", "Potential intervener"))




p <- ggplot(data = importance_df, aes(y = Gain, x = reorder(Feature, Gain), fill = type)) +
  geom_bar(stat = "identity", alpha = 1) +
  coord_flip() +
  th +
  labs(title = "Variable importance (XGBoost DART)", 
       y = "Gain score",
       x = "Feature",
       fill = "Type") +
  theme(legend.position = "bottom")
ggsave(file = "_output/_figures/xgb3_varimp.png", p, width = 15, height = 12, dpi = 300)

# Take 3: Compare models --------------------------------------------------

# saving models
models_take3 <- list(model_rf3, model_svm3, model_xgbdart3)
save(models_take3, file = "_output/_models/models_take3.RData")

models_compare3 <- resamples(list(RF=model_rf3, SVM=model_svm3, XGB=model_xgbdart3))

summary(models_compare3)

models_compare1_2_3 <- resamples(list(RF1=model_rf, RF2=model_rf2, RF3=model_rf3, SVM1=model_svm, SVM2=model_svm2, SVM3=model_svm3, XGB1=model_xgbdart, XGB2=model_xgbdart2, XGB3=model_xgbdart3))
summary(models_compare1_2_3)

# boxplots

models_compare3$values %>%
  select(1, ends_with("~Accuracy")) %>%
  gather(model, Accuracy, -1) %>%
  mutate(model = sub("~Accuracy", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Accuracy, y = model, fill = model)) -> p1

p1 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare3$values %>%
  select(1, ends_with("F1")) %>%
  gather(model, F1, -1) %>%
  mutate(model = sub("~Mean_F1", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = F1, y = model, fill = model)) -> p2

p2 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare3$values %>%
  select(1, ends_with("Precision")) %>%
  gather(model, Precision, -1) %>%
  mutate(model = sub("~Mean_Precision", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Precision, y = model, fill = model)) -> p3

p3 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

models_compare3$values %>%
  select(1, ends_with("Recall")) %>%
  gather(model, Recall, -1) %>%
  mutate(model = sub("~Mean_Recall", "", model)) %>%
  ggplot() +
  geom_boxplot(aes(x = Recall, y = model, fill = model)) -> p4

p4 + scale_y_discrete(limits = c("RF", "SVM", "XGB"))

plot_list <- list(p1, p2, p3, p4)

# putting the graphs together
model_comp_boxplots3 <- patchwork::wrap_plots(plot_list, ncol = 2, guides = 'collect') &
  theme(legend.position = 'bottom') & scale_fill_manual(values=c("#999999", "#E69F00", "#56B4E9")) & th & guides(fill = FALSE) &
  plot_annotation("Model performance across metrics", theme=theme(plot.title=element_text(hjust=0.1)))
ggsave(file = "_output/_figures/model_comp_boxplots3.png", model_comp_boxplots3, width = 8, height = 6, dpi = 300)

# let's graph F1 performance over iteration

f1_scores <- data.frame(
  model = c("RF", "RF", "RF", "SVM", "SVM", "SVM", "XGB", "XGB", "XGB"),
  iter = c("1", "2", "3", "1", "2", "3", "1", "2", "3"),
  f1_score = c(
    mean(model_rf$results$Mean_F1),
    mean(model_rf2$results$Mean_F1),
    mean(model_rf3$results$Mean_F1),
    mean(model_svm$results$Mean_F1, na.rm = TRUE),
    mean(model_svm2$results$Mean_F1, na.rm = TRUE),
    mean(model_svm3$results$Mean_F1, na.rm = TRUE),
    mean(model_xgbdart$results$Mean_F1),
    mean(model_xgbdart2$results$Mean_F1),
    mean(model_xgbdart3$results$Mean_F1)
  )
)
f1_scores

last_points <- f1_scores %>%
  group_by(model) %>%
  filter(row_number() == n())

# p <- ggplot(f1_scores, aes(x = iter, y = f1_score, color = model, group = model)) +
#   geom_line(size = 1.5) +
#   geom_text(data = last_points, aes(label = model), hjust = -0.2, size = 5) +
#   th +
#   scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
#   labs(title = "F1 Scores Across Models and Iterations",
#        x = "Iteration",
#        y = "Mean F1 Score",
#        fill = "Model") +
#   guides(color = FALSE) +
#   scale_x_discrete(limits = c(1, 2, 3), expand = c(0, 0))
# ggsave(file = "_output/_figures/model_comp_f1.png", p, width = 8, height = 6, dpi = 300)


p <- ggplot(f1_scores, aes(x = iter, y = f1_score, color = model, group = model)) +
  geom_line(size = 1.5) +
  geom_label(data = subset(f1_scores, iter == max(iter)),
             aes(label = model), hjust = 1.2, nudge_x = 0.1, show.legend = FALSE) +
  th +
  scale_color_manual(values=c("#999999", "#E69F00", "#56B4E9")) +
  labs(title = "F1 Scores Across Models and Iterations",
       x = "Iteration",
       y = "Mean F1 Score",
       fill = "Model") +
  guides(color = FALSE) +
  scale_x_discrete(limits = c(1, 2, 3), expand = c(0, 0)) +
  theme(axis.title.x = element_text( hjust = 0.47))
ggsave(file = "_output/_figures/model_comp_line_f1.png", p, width = 8, height = 6, dpi = 300)

# defining the order of models
model_order <- c("SVM", "RF", "XGB")

# convering model var to a factor with specified levels
f1_scores$model <- factor(f1_scores$model, levels = model_order)

# plot with specific order with the specified order
p <- ggplot(f1_scores, aes(x = iter, y = f1_score, fill = model, group = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  th +
  scale_fill_manual(values = c("#999999", "#E69F00", "#56B4E9")) +
  labs(title = "F1 Scores Across Models and Iterations",
       x = "Iteration",
       y = "Mean F1 Score",
       fill = "Model") +
  guides(color = FALSE)
ggsave(file = "_output/_figures/model_comp_bar_f1.png", p, width = 8, height = 6, dpi = 300)



# let's create confusion matrices

confusion_rf3 <- model_rf3[[5]]
filter(confusion_rf3, pred!=obs)

confusion_xgb3 <- model_xgbdart3[[5]]
# need to filter here, because I did savePredictions==TRUE on tune
confusion_xgb3 <- confusion_xgb3 %>%
  filter(nrounds == best_tuning$nrounds &
           max_depth == best_tuning$max_depth &
           eta == best_tuning$eta &
           gamma == best_tuning$gamma &
           subsample == best_tuning$subsample &
           colsample_bytree == best_tuning$colsample_bytree &
           rate_drop == best_tuning$rate_drop &
           skip_drop == best_tuning$skip_drop &
           min_child_weight == best_tuning$min_child_weight)
filter(confusion_xgb3, pred!=obs)

# let's look and compare

conf_rf3 <- table(confusion_rf3$obs, confusion_rf3$pred)
conf_rf3

conf_xgb3 <- table(confusion_xgb3$obs, confusion_xgb3$pred)
conf_xgb3
# just eye-balling, the performances look similar-ish, but xgb seems to confuse more
# xgb has more false positives, but fewer false negatives

# take a look over iterations

conf_rf1
conf_rf2
conf_rf3
# better accuracy, but false positives gone up as false negatives have gone down

conf_xgb1
conf_xgb2
conf_xgb3
# better accuracy, but false positives gone up while false negatives down
# more confusion over time

# # these results seem way off
# confusion_rf3 <- df.ml.train3 %>%
#   select(intervention)
# 
# confusion_rf3 <- bind_cols(confusion_rf3, prob.rf3) %>%
#   rename(neutral = 2,
#          gov= 3,
#          reb = 4)
# 
# confusion_rf3 <- confusion_rf3 %>%
#   mutate(pred = names(confusion_rf3[, 2:4])[max.col(confusion_rf3[, 2:4])])
# filter(confusion_rf3, is.na(pred))
# head(confusion_rf3)
# 
# confusion_rf3$pred <- factor(confusion_rf3$pred, levels = c('neutral', 'gov', 'reb'))
# 
# confusion_matrix <- table(confusion_rf3$intervention, confusion_rf3$pred)
# print(confusion_matrix)
# 
# # xgboost dart version
# confusion_xgb3 <- df.ml.train3 %>%
#   select(intervention)
# 
# confusion_xgb3 <- bind_cols(confusion_xgb3, prob.xgbdart3) %>%
#   rename(neutral = 2,
#          gov= 3,
#          reb = 4)
# 
# confusion_xgb3 <- confusion_xgb3 %>%
#   mutate(pred = names(confusion_xgb3[,2:4])[max.col(confusion_xgb3[,2:4], "first")])
# filter(confusion_xgb3, is.na(pred))
# head(confusion_xgb3)
# 
# confusion_xgb3$pred <- factor(confusion_xgb3$pred, levels = c('neutral', 'gov', 'reb'))
# 
# confusion_matrix <- table(confusion_xgb3$intervention, confusion_xgb3$pred)
# print(confusion_matrix)


# Take 3: RF model critique plot -----------------------------------------------------

df.rf.graph <- bind_cols(df.ml.train3, prob.rf3)

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
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,df.rf.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.5,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting government support (RF, take 3)"
)

ggsave(file = "_output/_figures/rf3_diagplot_gov.png", rfdiag_gov, width = 6, height = 6, dpi = 300)

# plot: reb
rfdiag_reb <- DiagPlot(
  f=df.rf.graph$reb,
  y=df.rf.graph$int_reb,
  labels= paste(df.rf.graph$target,df.rf.graph$intervener,df.rf.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.5,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting rebel support (RF, take 3)"
)

ggsave(file = "_output/_figures/rf3_diagplot_reb.png", rfdiag_reb, width = 6, height = 6, dpi = 300)

# Take 3: XGBoost DART model critique graph -------------------------------

df.xgbdart.graph <- bind_cols(df.ml.train3, prob.xgbdart3)

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
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener, df.xgbdart.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.55,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting government support (XGB, take 3)"
)

ggsave(file = "_output/_figures/xgb3_diagplot_gov.png", xgbdiag_gov, width = 6, height = 6, dpi = 300)

# plot: reb
xgbdiag_reb <- DiagPlot(
  f=df.xgbdart.graph$reb,
  y=df.xgbdart.graph$int_reb,
  labels= paste(df.xgbdart.graph$target,df.xgbdart.graph$intervener, df.xgbdart.graph$conflictID,sep='-'),
  worstN=10, # reduced in vain effort to make more readable
  label_spacing = 500, # had to up this substantially to make more readable, but still not working properly
  lab_adjust=.55,
  right_margin=10,
  top_margin=5,
  text_size=5,
  #  bw=bw,
  title="Predicting rebel support (XGB, take 3)"
)

ggsave(file = "_output/_figures/xgb3_diagplot_reb.png", xgbdiag_reb, width = 6, height = 6, dpi = 300)

# Take 3: Analyzing prediction errors -------------------------------------

# creating df to look at the worst predictions
rf3_worst_gov <- df.rf.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_gov-gov) %>% 
  arrange(gov) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_gov) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

rf3_worst_gov <- rf3_worst_gov %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

rf3_worst_reb <- df.rf.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_reb-reb) %>% 
  arrange(reb) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_reb) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

rf3_worst_reb <- rf3_worst_reb %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

# takeaway: Ukraine gov-sided is still an issue, with some improvement on specific interveners
# takeaway: Better predictions on Myanmar gov-sided
# takeaway: Better on Myanmar reb-sided various conflicts
# takeaway: Ethiopia still an issue on reb-sided

# now look at xgbdart
xgbdart3_worst_gov <- df.xgbdart.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_gov-gov) %>% 
  arrange(gov) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_gov) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

xgbdart3_worst_gov <- xgbdart3_worst_gov %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

xgbdart3_worst_reb <- df.xgbdart.graph %>%
  select(intervention, year, conflictID, ccode1, target, ccode2, intervener, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(y_minus_f=int_reb-reb) %>% 
  arrange(reb) %>% 
  mutate(forecastOrder = row_number()) %>%
  group_by(int_reb) %>% 
  arrange(desc(abs(y_minus_f))) %>%
  mutate(abserr=abs(y_minus_f)) %>% 
  mutate(isworstn=ifelse(row_number()<=10, 1, 0)) %>%
  filter(isworstn==1)

xgbdart3_worst_reb <- xgbdart3_worst_reb %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==0 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(dyads = paste(unique(dyad_name), collapse = ", "),
                        gov_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID") %>%
  ungroup() %>%
  left_join(ucdpexternal %>% filter(actor_nonstate==1 & ext_alleged==0) %>%
              group_by(conflictID) %>%
              summarise(reb_supporters = paste(unique(ext_name), collapse = ", ")),
            by = "conflictID")

# takeaway: fewer confusions on gov-sided
# takeaway: Ethiopia still an issue on gov-sided

## GRAPHING

# let's look at error over time
rf3_graph_worst <- df.rf.graph %>%
  select(intervention, year, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(error_gov=abs(int_gov-gov)) %>%
  mutate(error_reb=abs(int_reb-reb)) 

ggplot(rf3_graph_worst, aes(x = year, y = error_gov)) +
  geom_point()

ggplot(rf3_graph_worst, aes(x = year, y = error_reb)) +
  geom_point()

xgbdart3_graph_worst <- df.xgbdart.graph %>%
  select(intervention, year, neutral, gov, reb, int_neutral, int_gov, int_reb) %>%
  mutate(error_gov=abs(int_gov-gov)) %>%
  mutate(error_reb=abs(int_reb-reb)) 

ggplot(xgbdart3_graph_worst, aes(x = year, y = error_gov)) +
  geom_point()

ggplot(xgbdart3_graph_worst, aes(x = year, y = error_reb)) +
  geom_point()

# takeaway: some correlation over time, but seems less than take 2

# let's look at density of errors by outcome
# gov error first
ggplot(rf3_graph_worst, aes(error_gov)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)
# then reb error
ggplot(rf3_graph_worst, aes(error_reb)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)

# let's look at density of errors by outcome
# gov error first
ggplot(xgbdart3_graph_worst, aes(error_gov)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)
# then reb error
ggplot(xgbdart3_graph_worst, aes(error_reb)) +
  geom_density(aes(fill=factor(intervention)), alpha = 0.8)


# Take 3: Cases across models ---------------------------------------------

# starting by looking at worst predicted conflicts from take 1 and see how they've improved



### RF

graph_cases <- df.ml.train3 %>%
  select(intervention)

# putting together the predicted probabilities across three RF iterations
graph_cases <- bind_cols(graph_cases, trainID, prob.rf, prob.rf2, prob.rf3) %>%
  rename(p_neutral_1 = 6,
         p_gov_1 = 7,
         p_reb_1 = 8,
         p_neutral_2 = 9,
         p_gov_2 = 10,
         p_reb_2 = 11,
         p_neutral_3 = 12,
         p_gov_3 = 13,
         p_reb_3 = 14)

graph_cases <- graph_cases %>% 
  mutate(int_neutral = ifelse(intervention=="neutral", 1, 0)) %>%
  mutate(int_gov = ifelse(intervention=="gov", 1, 0)) %>%
  mutate(int_reb = ifelse(intervention=="rebel", 1, 0)) %>%
  mutate(target = countrycode(ccode1, "cown", "country.name")) %>%
  mutate(intervener = countrycode(ccode2, "cown", "country.name"))

labels <- c("gov", "reb")

# for loop to create the absolute error vars across gov and reb and model iterations
for (label in labels) {
  for (i in 1:3) {
    abs_error_col <- paste0("abs_error_", label, "_", i)
    int_col <- paste0("int_", label)
    p_col <- paste0("p_", label, "_", i)
    
    graph_cases <- graph_cases %>%
      mutate(!!abs_error_col := abs(.data[[int_col]] - .data[[p_col]])) # the bang-bang lets me dynamically insert var name from above
  }
}

# creating df for top 10 worst
worst_cases_gov <- graph_cases %>%
  group_by(target, conflictID) %>%
  summarise(sum_error_gov_1 = sum(abs_error_gov_1), 
            sum_error_gov_2 = sum(abs_error_gov_2), 
            sum_error_gov_3 = sum(abs_error_gov_3)) %>%
  ungroup() %>%
  arrange(desc(sum_error_gov_1)) %>%
  slice_head(n = 10) %>%
  pivot_longer(
    cols = starts_with("sum_error_gov_"),
    names_to = "iteration",
    names_prefix = "sum_error_gov_",
    values_to = "sum_error",
    values_drop_na = TRUE) %>%
  mutate(iteration = as.numeric(iteration)) %>%
  mutate(lab = paste0(target, " ", "(", conflictID, ")"))

head(worst_cases_gov, n = 10)

p1 <- ggplot(worst_cases_gov, aes(x = iteration, y = sum_error, color = as.factor(conflictID)), show.legend = FALSE) +
  geom_line() +
  geom_label(data = subset(worst_cases_gov, iteration == max(iteration)),
             aes(label = lab), hjust = 1.2, nudge_x = 0.1, show.legend = FALSE) +
  guides(color = FALSE) +
  scale_x_continuous(breaks = seq(min(worst_cases_gov$iteration), max(worst_cases_gov$iteration), by = 1)) +
  ggtitle("Worst predicted government support by conflict using Random Forest") +
  labs(y = "Absolute error",
      x = "Iteration") +
  th +
  scale_x_discrete(limits = c(1, 2, 3), expand = c(0, 0))
ggsave(file = "_output/_figures/worst_rf_gov.png", p1, width = 8, height = 6, dpi = 300)

# an alternative solution
# library(ggrepel)
# 
# ggplot(worst_cases_gov, aes(x = iteration, y = sum_error, color = as.factor(conflictID))) +
#   geom_line() +
#   geom_text_repel(data = subset(worst_cases_gov, iteration == max(iteration)),
#                   aes(label = lab), nudge_x = 0.2, direction = "x", show.legend = FALSE) +
#   guides(color = FALSE) + # This removes the legend for color
#   scale_x_continuous(breaks = seq(min(worst_cases_gov$iteration), max(worst_cases_gov$iteration), by = 1))

# time to look at rebel prediction error

worst_cases_reb <- graph_cases %>%
  group_by(target, conflictID) %>%
  summarise(sum_error_reb_1 = sum(abs_error_reb_1), 
            sum_error_reb_2 = sum(abs_error_reb_2), 
            sum_error_reb_3 = sum(abs_error_reb_3)) %>%
  ungroup() %>%
  arrange(desc(sum_error_reb_1)) %>%
  slice_head(n = 10) %>%
  pivot_longer(
    cols = starts_with("sum_error_reb_"),
    names_to = "iteration",
    names_prefix = "sum_error_reb_",
    values_to = "sum_error",
    values_drop_na = TRUE) %>%
  mutate(iteration = as.numeric(iteration)) %>%
  mutate(lab = paste0(target, " ", "(", conflictID, ")"))

head(worst_cases_reb, n = 10)

p2 <- ggplot(worst_cases_reb, aes(x = iteration, y = sum_error, color = as.factor(conflictID)), show.legend = FALSE) +
  geom_line() +
  geom_label(data = subset(worst_cases_reb, iteration == max(iteration)),
             aes(label = lab), hjust = 1.2, nudge_x = 0.1, show.legend = FALSE) +
  guides(color = FALSE) +
  scale_x_continuous(breaks = seq(min(worst_cases_reb$iteration), max(worst_cases_reb$iteration), by = 1)) +
  ggtitle("Worst predicted rebel support by conflict using Random Forest") +
  labs(y = "Absolute error",
       x = "Iteration") +
  th +
  scale_x_discrete(limits = c(1, 2, 3), expand = c(0, 0))
ggsave(file = "_output/_figures/worst_rf_reb.png", p2, width = 8, height = 6, dpi = 300)



### XGBoost DART

graph_cases <- df.ml.train3 %>%
  select(intervention)

# putting together the predicted probabilities across three RF iterations
graph_cases <- bind_cols(graph_cases, trainID, prob.xgbdart, prob.xgbdart2, prob.xgbdart3) %>%
  rename(p_neutral_1 = 6,
         p_gov_1 = 7,
         p_reb_1 = 8,
         p_neutral_2 = 9,
         p_gov_2 = 10,
         p_reb_2 = 11,
         p_neutral_3 = 12,
         p_gov_3 = 13,
         p_reb_3 = 14)

graph_cases <- graph_cases %>% 
  mutate(int_neutral = ifelse(intervention=="neutral", 1, 0)) %>%
  mutate(int_gov = ifelse(intervention=="gov", 1, 0)) %>%
  mutate(int_reb = ifelse(intervention=="rebel", 1, 0)) %>%
  mutate(target = countrycode(ccode1, "cown", "country.name")) %>%
  mutate(intervener = countrycode(ccode2, "cown", "country.name"))

labels <- c("gov", "reb")

# for loop to create the absolute error vars across gov and reb and model iterations
for (label in labels) {
  for (i in 1:3) {
    abs_error_col <- paste0("abs_error_", label, "_", i)
    int_col <- paste0("int_", label)
    p_col <- paste0("p_", label, "_", i)
    
    graph_cases <- graph_cases %>%
      mutate(!!abs_error_col := abs(.data[[int_col]] - .data[[p_col]])) # the bang-bang lets me dynamically insert var name from above
  }
}

worst_cases_gov <- graph_cases %>%
  group_by(target, conflictID) %>%
  summarise(sum_error_gov_1 = sum(abs_error_gov_1), 
            sum_error_gov_2 = sum(abs_error_gov_2), 
            sum_error_gov_3 = sum(abs_error_gov_3)) %>%
  ungroup() %>%
  arrange(desc(sum_error_gov_1)) %>%
  slice_head(n = 10) %>%
  pivot_longer(
    cols = starts_with("sum_error_gov_"),
    names_to = "iteration",
    names_prefix = "sum_error_gov_",
    values_to = "sum_error",
    values_drop_na = TRUE) %>%
  mutate(iteration = as.numeric(iteration)) %>%
  mutate(lab = paste0(target, " ", "(", conflictID, ")"))

head(worst_cases_gov, n = 10)

p3 <- ggplot(worst_cases_gov, aes(x = iteration, y = sum_error, color = as.factor(conflictID)), show.legend = FALSE) +
  geom_line() +
  geom_label(data = subset(worst_cases_gov, iteration == max(iteration)),
             aes(label = lab), hjust = 1.2, nudge_x = 0.1, show.legend = FALSE) +
  guides(color = FALSE) +
  scale_x_continuous(breaks = seq(min(worst_cases_gov$iteration), max(worst_cases_gov$iteration), by = 1)) +
  ggtitle("Worst predicted government support by conflict using XGBoost DART") +
  labs(y = "Absolute error",
       x = "Iteration") +
  th +
  scale_x_discrete(limits = c(1, 2, 3), expand = c(0, 0))
ggsave(file = "_output/_figures/worst_xgbdart_gov.png", p3, width = 8, height = 6, dpi = 300)

# rebel prediction error

worst_cases_reb <- graph_cases %>%
  group_by(target, conflictID) %>%
  summarise(sum_error_reb_1 = sum(abs_error_reb_1), 
            sum_error_reb_2 = sum(abs_error_reb_2), 
            sum_error_reb_3 = sum(abs_error_reb_3)) %>%
  ungroup() %>%
  arrange(desc(sum_error_reb_1)) %>%
  slice_head(n = 10) %>%
  pivot_longer(
    cols = starts_with("sum_error_reb_"),
    names_to = "iteration",
    names_prefix = "sum_error_reb_",
    values_to = "sum_error",
    values_drop_na = TRUE) %>%
  mutate(iteration = as.numeric(iteration)) %>%
  mutate(lab = paste0(target, " ", "(", conflictID, ")"))

head(worst_cases_reb, n = 10)

p4 <- ggplot(worst_cases_reb, aes(x = iteration, y = sum_error, color = as.factor(conflictID)), show.legend = FALSE) +
  geom_line() +
  geom_label(data = subset(worst_cases_reb, iteration == max(iteration)),
             aes(label = lab), hjust = 1.2, nudge_x = 0.1, show.legend = FALSE) +
  guides(color = FALSE) +
  scale_x_continuous(breaks = seq(min(worst_cases_reb$iteration), max(worst_cases_reb$iteration), by = 1)) +
  ggtitle("Worst predicted rebel support by conflict using XGBoost DART") +
  labs(y = "Absolute error",
       x = "Iteration") +
  th +
  scale_x_discrete(limits = c(1, 2, 3), expand = c(0, 0))
ggsave(file = "_output/_figures/worst_xgbdart_reb.png", p4, width = 8, height = 6, dpi = 300)

# #putting them all together
# 
# plot_list <- list(p1, p2, p3, p4)
# 
# # putting the graphs together
# p <- patchwork::wrap_plots(plot_list, ncol = 2, guides = 'collect') &
#   theme(legend.position = 'bottom')
# ggsave(file = "_output/_figures/worst_conf.pdf", p, width = 8, height = 6, dpi = 300)

# creating df for absolute errors for all
df_pred <- graph_cases %>%
  group_by(target, conflictID) %>%
  summarise(sum_error_gov_1 = sum(abs_error_gov_1), 
            sum_error_gov_2 = sum(abs_error_gov_2), 
            sum_error_gov_3 = sum(abs_error_gov_3)) %>%
  ungroup() %>%
  pivot_longer(
    cols = starts_with("sum_error_gov_"),
    names_to = "iteration",
    names_prefix = "sum_error_gov_",
    values_to = "sum_error",
    values_drop_na = TRUE) %>%
  mutate(iteration = as.numeric(iteration)) %>%
  mutate(lab = paste0(target, " ", "(", conflictID, ")"))
summary(df_pred)
head(df_pred)

p <- ggplot(df_pred, aes(x = iteration, y = sum_error, group = conflictID), show.legend = FALSE) +
  geom_line(color = "red", alpha = 0.5) +
  scale_x_continuous(breaks = seq(min(worst_cases_gov$iteration), max(worst_cases_gov$iteration), by = 1)) +
  ggtitle("Worst predicted government support by conflict using XGBoost DART") +
  labs(y = "Absolute error",
       x = "Iteration") +
  th +
  scale_x_discrete(limits = c(1, 2, 3), expand = c(0, 0))
ggsave(file = "_output/_figures/conf_abserror.png", p, width = 8, height = 6, dpi = 300)





# Prototyping graphs ------------------------------------------------------

## partial dependence plots

pdp_output <- partial(model_xgbdart3, pred.var = "mindist", train = df.ml.train3, grid.resolution = 20)
plot(pdp_output)

# looking at some specific countries with conflicts where models perform poorly
ethiopia_conflicts <- unique(df.rf.graph$conflictID[df.rf.graph$target=="Ethiopia"])

myanmar_conflicts <- unique(df.rf.graph$conflictID[df.rf.graph$target=="Myanmar (Burma)"])

ukraine_conflicts <- unique(df.rf.graph$conflictID[df.rf.graph$target=="Ukraine"])


# prototype: ethiopia gov
prototype <- graph_cases %>%
  filter(conflictID %in% ethiopia_conflicts) %>%
  select(3, 4, starts_with("abs_error_gov_")) %>%
  mutate(inter_conf = paste0(ccode2, "-", conflictID)) %>%
  select(inter_conf, everything()) %>%
  pivot_longer(
    cols = starts_with("abs_error_gov_"),
    names_to = "iteration",
    names_prefix = "abs_error_gov_",
    values_to = "abs_error",
    values_drop_na = TRUE) %>%
  mutate(iteration = as.numeric(iteration))

ggplot(prototype, aes(x = iteration, y = abs_error, by = as.factor(ccode2), color = as.factor(conflictID))) +
  geom_line()

prototype2 <- prototype %>%
  group_by(conflictID, iteration) %>%
  summarise(sum_error = sum(abs_error_gov)) %>%
  ungroup()

ggplot(prototype2, aes(x = iteration, y = sum_error, color = as.factor(conflictID))) +
  geom_line() #+
  #geom_area(aes(fill = as.factor(conflictID), group = as.factor(conflictID)),
  #          alpha = 0.5, position = 'identity')

# prototype: ethiopia reb
prototype <- graph_cases %>%
  filter(conflictID %in% ethiopia_conflicts) %>%
  select(3, 4, starts_with("abs_error_reb_")) %>%
  mutate(inter_conf = paste0(ccode2, "-", conflictID)) %>%
  select(inter_conf, everything()) %>%
  pivot_longer(
    cols = starts_with("abs_error_reb_"),
    names_to = "iteration",
    names_prefix = "abs_error_reb_",
    values_to = "abs_error",
    values_drop_na = TRUE) %>%
  mutate(iteration = as.numeric(iteration))

ggplot(prototype, aes(x = iteration, y = abs_error, by = as.factor(ccode2), color = as.factor(conflictID))) +
  geom_line()

prototype2 <- prototype %>%
  group_by(conflictID, iteration) %>%
  summarise(sum_error = sum(abs_error)) %>%
  ungroup()

ggplot(prototype2, aes(x = iteration, y = sum_error, color = as.factor(conflictID))) +
  geom_line()


# Take 4: XGBoost DART re-tuning ------------------------------------------

plot(model_xgbdart3)

tune_grid_p <- expand.grid(
  nrounds = c(150, 200, 250), # number of boosting rounds, cutting 100
  max_depth = c(3, 6, 9), # max depth of trees
  eta = c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3), # default learning rate, how much each new tree contributes to final model, low to prevent overfitting
  gamma = 1, # leaving low because elnet suggests no regularization; might need to increase to make simpler trees and prevent overfitting
  subsample = 0.8,
  colsample_bytree = seq(0.5, 1, 0.1), # reducing complexity of each tree
  rate_drop = 0.1,
  skip_drop = 0.5,
  min_child_weight = c(0, 5, 10) # might increase to prevent overfitting
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

# this took ~94h on my mbp to run
model_xgbdart4 <- train(
  intervention ~ .,
  data = df.ml.train3, # using same data as previous iteration
  method = 'xgbDART',
  tuneGrid = tune_grid_p,
  trControl = fitControl_p,
  metric = "Mean_F1",
  verbose = FALSE,
  nthread = 1
)

#stopCluster(num_cores)
stopImplicitCluster()

# saving model
models_take4 <- list(model_xgbdart4)
save(models_take4, file = "_output/_models/models_take4.RData")

# pulling out tuning results for graphs (might wanna do a df with all hyperparameters using do.call())
tuning_results <- model_xgbdart4$results

ggplot(tuning_results, aes(x= Mean_F1)) +
  geom_density(alpha = 0.7, fill = "lightblue") +
  th

hyper <- c('nrounds', 'max_depth', 'eta', 'colsample_bytree', 'min_child_weight')
labels <- c('No. of rounds', 'Maximum depth', 'Learning rate', 'Subsampling of columns', 'Min. child weight')

xgboost4_tune <- lapply(seq_along(hyper), function(i) {
  ggplot(tuning_results, aes(x = !!sym(hyper[i]), y = mean_f1_values, fill = factor(!!sym(hyper[i])))) +
    geom_violin(alpha = 0.7) +
    coord_flip() +  
    labs(x = labels[i],
         y = "") +
    th +
    guides(fill = FALSE) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})

xgb3_tune_graphs <- patchwork::wrap_plots(xgboost4_tune, ncol = 5, guides = 'collect') &
  theme(legend.position = 'bottom') &
  plot_annotation("XGBoost DART model performance (Mean F1) vs. hyperparameters", theme=theme(plot.title=element_text(hjust=0.1)))

ggsave(file = "_output/_figures/xgb4_tuning_violin.png", xgb3_tune_graphs, width = 8, height = 6, dpi = 300)



ggplot(df, aes(x = hyp_value, y = mean_f1_values, fill = factor(hyp_value))) +
  geom_violin(alpha = 0.7) +
  coord_flip() +
  labs(title = paste("Violin plot of Mean F1 vs.", hyperparameter),
       x = hyperparameter, y = "Mean F1") +
  th +
  guides(fill = FALSE)

ggplot(tuning_results, aes(x = eta, y = Mean_F1, fill = factor(eta))) +
  geom_violin(alpha = 0.7) +
  coord_flip() +
  labs(title = paste("Violin plot of Mean F1 vs.", hyperparameter),
       x = hyperparameter, y = "Mean F1") +
  th +
  guides(fill = FALSE)


# need to add in learning rate, at least, and maybe subsample and lambda and alpha
# can tighten boosting iterations and max_depth, cut bottom values

# Test pre-processing -----------------------------------------------------------

# # THIS WILL NEED TO BE UPDATED FOR TAKE 3
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
