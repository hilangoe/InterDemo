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
#library(fastAdaboost) #not available anymore, but need it for Adaboost algo

# NOTES ON RE-DO:
# use createDataPartition to ensure proportion of Y
# do partition separately up front
# descriptive statistics with skimr::skim_to_wide(trainData)
# imputation with preProcess()
# dummy variables using dummyVars() (no implicit base category!)
# other pre-processing required, like transformation
# feature selection through rfe (not necessary here)


# Trim main dataframe -----------------------------------------------------

# df.int is the main dataframe, which includes observations where T supports both sides starting in same year

# dropping irrelevant variables
df.ml <- df.int %>%
  select(., -c("rivalryno", "rivalryname", "styear", "endyear", "govsupport", "rebsupport", "govsupportyear", "rebsupportyear", "bothsides", "geo_NA", "region", "type1", "type2", "type3", 'govrec', 'rebrec', 'conflictID', 'ccode1', 'ccode2')) %>%
  select(intervention, everything())
# moving DV first in df

# dropping observations with intervention on both sides in the same year
df.ml <- df.ml[df.ml$intervention!=3, ]
df.ml$intervention <- factor(df.ml$intervention,
                                        levels = c(0, 1, 2))

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

skimmed[72:75,] # might not be correct final column number

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

## gonna skip over the recursive feature elimination (rfe) stuff here


# Training RF model -------------------------------------------------------

modelLookup('rf')

set.seed(111)

#model_rf = train(intervention ~ ., data=df.ml.train, method='rf')
# this is insanely slow

# doing some hyper parameter tuning to speed things up

fitControl <- trainControl(
  method = 'cv', #k-fold cross validation
  number = 5, # this should speed things up
  savePredictions = 'final',
#  classProbs = T, # I want predictions rather than probabilities
  summaryFunction = multiClassSummary
)

model_rf = train(intervention ~ ., data=df.ml.train, method='rf', tuneLength = 5, trControl = fitControl)
# had some trouble with ROC here as metric, so using default instead

fitted.rf <- predict(model_rf)

# looking at variable importance

varimp_rf <- varImp(model_rf)

plot(varimp_rf, main = "Variable importance with Random Forest")


# Preparing testing df and testing ---------------------------------------------------

# impute missing data
df.ml.test2 <- predict(missing_model, df.ml.test)

# transform vars
df.ml.test3 <- predict(range_model, df.ml.test2)

predicted_rf <- predict(model_rf, df.ml.test3)
head(predicted_rf)

# Confusion matrix
confusionMatrix(reference = df.ml.test3$intervention, data = predicted_rf, mode='everything', positive='MM')
# performs ok on false positives (precision), but poorly on false negatives (recall)
# better at predicting gov intervention than reb intervention

# Tuning ------------------------------------------------------------------

# had to do this earlier to get model to estimate in a timely manner
# maybe try tuneGrid instead of tuneLength

# Training alternative models ---------------------------------------------------

## elnet (glmnet)

model_elnet = train(intervention ~ ., 
                    data=df.ml.train, 
                    method='glmnet', 
#                    tuneLength = 5,
                    tuneGrid = expand.grid(
                      alpha = 0:1, 
                      lambda = seq(0, 15, by = 0.01)), #upping penalty here to see if it improves performance
                    maxit = 100000,
                    trControl = fitControl)
model_elnet
# this is still causing issues with convergence without the tuneGrid specified

predicted_elnet <- predict(model_elnet, df.ml.test3)
head(predicted_elnet)

confusionMatrix(reference = df.ml.test3$intervention, data = predicted_elnet, mode='everything', positive='MM')
# really poor performance: need to tune on lambda

## svm

model_svm = train(intervention ~ ., 
                    data=df.ml.train, 
                    method='svmRadial', 
                    tuneLength = 15,
                    maxit = 100000,
                    trControl = fitControl)
model_svm

predicted_svm <- predict(model_svm, df.ml.test3)
head(predicted_svm)

confusionMatrix(reference = df.ml.test3$intervention, data = predicted_svm, mode='everything', positive='MM')

# Comparing models --------------------------------------------------------

models_compare <- resamples(list(RF=model_rf, ELNET=model_elnet, SVM=model_svm))

summary(models_compare)

scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)
# the statistics here on mean precision are very misleading
