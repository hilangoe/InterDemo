# script for tree and random forest analysis predicting civil war intervention
# analysis here is first cut, results presented at ISA 2023
# upshot: models not performing well at all, huge classification error rate, lot of false negatives

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

# df.int is the main dataframe, which includes observations where T supports both sides starting in same year

head(df.int)
# lot of variables here


# Tree --------------------------------------------------------------------

### Creating a classification tree

# step 1: define a tree
# drop: rivalryno rivalryname styear endyear govsupport rebsupport govsupportyear rebsupportyear bothsides geo_NA

df.int.tree <- df.int %>%
  select(., -c("rivalryno", "rivalryname", "styear", "endyear", "govsupport", "rebsupport", "govsupportyear", "rebsupportyear", "bothsides", "geo_NA", "region", "type1", "type2", "type3"))

df.int.tree$intervention <- factor(df.int.tree$intervention)

# need to subtract ccode1 ccode2

tree.int <- tree(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, df.int.tree, mindev=0.0001, minsize=50)

# step 2: summary

summary(tree.int)

# step 3: plot the tree

plot(tree.int, type="uniform")

# step 4: text tree
text(tree.int)

summary(tree.int)

# step 5: evaluate performance of tree

set.seed(2)

attach(df.int.tree)

int.train <- sample(1:nrow(df.int.tree), 10000)
int.test <- df.int.tree[-int.train, ]

intervention.test <- intervention[-int.train]
tree.int.subset <- tree(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, df.int.tree, mindev=0.0001, minsize=50, subset = int.train)

tree.predict <- predict(tree.int.subset, int.test, type = "class")

table(tree.predict, intervention.test)
(15762+27)/16202

# not sure why I'm getting an n larger than the training set
# while the test error is low, that's mostly because this is a zero-inflated df

# step 6: pruning

set.seed(7)
cv.int <- cv.tree(tree.int, FUN = prune.misclass)

names(cv.int)
cv.int

par(mfrow = c(1, 2))
plot(cv.int$size, cv.int$dev, type = "b")
plot(cv.int$k, cv.int$dev, type = "b")

prune.int <- prune.misclass(tree.int, best = 28)
plot(prune.int)
text(prune.int, pretty = 0)
# screeengrabbed this for the paper

tree.predict <- predict(prune.int, int.test, type = "class")
table(tree.predict, intervention.test)
(15808+38+7)/16202
# performs slightly better, and the tree is a lot easier to interpret
# i wonder if the overly inclusive sample of potential interveners makes this too cumbersome


# RF (sparse) -------------------------------------------------------------

## on to random forest
# need to partition again here

set.seed(3)

bag.int <- randomForest(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear, data = df.int.tree, subset = int.train, mtry = 21, importance = TRUE, na.action=na.exclude)
bag.int
# seems like high class error on the positive outcomes

plot(bag.int)

varImpPlot(bag.int)

# this is building off of the regression trees, which I haven't done so far
yhat.bag <- predict(bag.int, newdata = df.int.tree[-int.train, ])

# this is not appropriate
#plot(yhat.bag, intervention.test)
#abline(0,1)

# this is better
table(yhat.bag, intervention.test)
(11596+43+2)/11946
# actually performs worse

## adding in more variables
set.seed(3)

bag.int2 <- randomForest(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear + polydist + v2x_polyarchy1 + v2x_polyarchy2 + libdist + v2x_liberal1 + v2x_liberal2 + libdemdist + v2x_libdem1 + v2x_libdem2 + opendist + kaopen1 + kaopen2 + kappavv, data = df.int.tree, subset = int.train, mtry = 30, importance = TRUE, na.action=na.exclude)
bag.int2

plot(bag.int2)

varImpPlot(bag.int2)


# RF (full RHS) -----------------------------------------------------------

### TO-DO: drop some covars that aren't appropriate

### RF on full RHS, but have to cut outcome 3

df.int.tree.full <- df.int.tree[df.int.tree$intervention!=3, ]
df.int.tree.full$intervention <- factor(df.int.tree.full$intervention,
                                    levels = c(0, 1, 2))

print("Count of missing values by column wise")
missing <- sapply(df.int.tree.full, function(x) sum(is.na(x)))
df.missing <- as.data.frame(missing)
df.missing <- df.missing %>% arrange(desc(missing))
print(df.missing)

# create index of train and test sets
int.train.full <- sample(1:nrow(df.int.tree.full), 10000)
int.test.full <- df.int.tree.full[-int.train.full, ]
# should just create new dataframes here, so imputation is cleaner too, no leakage

intervention.test.full <- df.int.tree.full$intervention[-int.train.full]

bag.int.full <- randomForest(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear + polydist + v2x_polyarchy1 + v2x_polyarchy2 + libdist + v2x_liberal1 + v2x_liberal2 + libdemdist + v2x_libdem1 + v2x_libdem2 + opendist + kaopen1 + kaopen2 + kappava + kappavv + cow_defense + cow_entente + cow_neutral + cow_nonagg + flow1 + flow2 + v2x_polyarchy1 + v2x_polyarchy2 + v2x_libdem1 + v2x_libdem2 + v2x_liberal1 + v2x_liberal2 + kaopen1 + kaopen2 + geo, data = df.int.tree.full, subset = int.train.full, mtry = 30, importance = TRUE, na.action=na.roughfix)
bag.int.full
# still too many NAs
# using rough fix here, but might try imputation

# predict on test data using trained model
yhat.bag.full <- predict(bag.int.full, newdata = df.int.tree.full[-int.train.full, ])

# plotting predictions
table(yhat.bag.full, intervention.test.full)

plot(bag.int.full)

varImpPlot(bag.int.full)



# RF (all RHS) ------------------------------------------------------------

df.int.all <- df.int.tree.full %>% select(-c('govrec', 'rebrec', 'conflictID', 'ccode1', 'ccode2'))

int.train.all <- sample(1:nrow(df.int.all), 10000)
int.test.all <- df.int.all[-int.train.all, ]

intervention.test.all <- df.int.all$intervention[-int.train.all]

bag.int.all <- randomForest(intervention ~ ., data = df.int.all, subset = int.train.all, mtry = 30, importance = TRUE, na.action=na.roughfix)
bag.int.all
# still too many NAs
# using rough fix here, but might try imputation

# predict on test data using trained model
yhat.bag.all <- predict(bag.int.all, newdata = df.int.all[-int.train.all, ])
# this produces a lot of missing data

# plotting predictions
table(yhat.bag.all, intervention.test.all)
# why does this produce so many NAs?

plot(bag.int.all)

varImpPlot(bag.int.all)


# RF (imputation) ---------------------------------------------------------

# imputation:
# gonna use df.int.all so we leave out the nonsensical variables
set.seed(111)
int.train.all.imp <- sample(1:nrow(df.int.all), 18339)
int.test.all.imp <- df.int.all[-int.train.all.imp, ]

# pulling out outcomes from test data
int.imp <- df.int.all$intervention[-int.train.all.imp]

set.seed(222)
df.rf.imp <- rfImpute(intervention ~ ., 
                      data = df.int.all, subset = int.train.all.imp) # question is if impute on whole df or just train df
set.seed(333)
rf.imp <- randomForest(intervention ~ ., 
                       df.rf.imp, ntry = 300, mtry = 30, importance = TRUE)
print(rf.imp)
# need to impute test data too!!!
df.rf.imp.test <- rfImpute(intervention ~ ., 
                      data = int.test.all.imp)

yhat.imp <- predict(rf.imp, newdata = df.rf.imp.test)

# plotting predictions
table(yhat.imp, int.imp)

plot(rf.imp)

pdf(file = "_output/_figures/varimpplot1.pdf")
varImpPlot(rf.imp)
dev.off()

# RF (all training) -------------------------------------------------------

# using the whole dataset here to see if that helps with the classification error
set.seed(777)
rf.alltrain <- randomForest(intervention ~ ., 
                       df.int.all, ntry = 300, mtry = 30, importance = TRUE, na.action=na.roughfix)
print(rf.alltrain)

plot(rf.alltrain)

pdf(file = "_output/_figures/varimpplot2.pdf")
varImpPlot(rf.alltrain)
dev.off()


# RF (min distance) -------------------------------------------------------

## smaller sample by restricting min distance

set.seed(3)

df.int.tree3 <- df.int.tree[df.int.tree$mindist<2500, ]
df.int.tree3 <- df.int.tree3[df.int.tree3$intervention!=3, ]
df.int.tree3$intervention <- factor(df.int.tree3$intervention,
                                     levels = c(0, 1, 2))

int.train3 <- sample(1:nrow(df.int.tree3), 1000)
int.test3 <- df.int.tree3[-int.train3, ]

intervention.test3 <- df.int.tree3$intervention[-int.train3]

bag.int3 <- randomForest(intervention ~ p2dist + polity21 + polity22 + mindist + ongoingrivalry + cowmaj1 + cowmaj2 + wbgdp2011est1 + wbgdp2011est2 + wbpopest1 + wbpopest2 + wbgdppc2011est1 + wbgdppc2011est2 + upop_mc_1 + upop_mc_2 + cinc_mc_1 + cinc_mc_2 + growth_wdi_1 + growth_wdi_2 + acdcwyear + acdiwyear + polydist + v2x_polyarchy1 + v2x_polyarchy2 + libdist + v2x_liberal1 + v2x_liberal2 + libdemdist + v2x_libdem1 + v2x_libdem2 + opendist + kaopen1 + kaopen2 + kappavv, data = df.int.tree3, subset = int.train3, mtry = 30, importance = TRUE, na.action=na.exclude)
bag.int3

plot(bag.int3)

varImpPlot(bag.int3)
# now polydist is not important, but kappavv is. so the closer you are, the more important policy is over ideology (but that might be because of spatial clustering of polities)
