# Rare Events - SMOTE






### Initializations
Loading some necessary packages as well as specifying a number of cores for parallel to expedite computation and setting a seed for reproducibility of the results.


```r
libs <- c("DMwR","caret","pROC","doParallel")
lapply(libs, require, character.only=TRUE)
registerDoParallel(6)
set.seed(1234)
```


### Data Preparation

```r
## Downlood data set
hyper <-read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data', header=F)
names <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names', header=F, sep='\t')[[1]]

## Clean names <- gsub(pattern =":|[.]",x = names, replacement="")
colnames(hyper) <- names
colnames(hyper) <-c("target", "age", "sex", "on_thyroxine", "query_on_thyroxine",
                    "on_antithyroid_medication", "thyroid_surgery", "query_hypothyroid",
                    "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium",
                    "goitre", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured",
                    "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured",
                    "TBG")

## Clean up observation values
hyper$target <- ifelse(hyper$target=='negative',0,1)
ind <- sapply(hyper, is.factor)
hyper[ind] <- lapply(hyper[ind], as.character)
hyper[ hyper == "?" ] = NA
hyper[ hyper == "f" ] = 0
hyper[ hyper == "t" ] = 1
hyper[ hyper == "n" ] = 0
hyper[ hyper == "y" ] = 1
hyper[ hyper == "M" ] = 0
hyper[ hyper == "F" ] = 1
hyper[ind] <- lapply(hyper[ind], as.numeric)
repalceNAsWithMean <- function(x) {replace(x, is.na(x), mean(x[!is.na(x)]))}
hyper <- repalceNAsWithMean(hyper)
hyper$target <- as.factor(ifelse(hyper$target == 1, "positive", "negative"))
hyper <- na.omit(hyper)
```

We are testing methods to deal with rare events, so just how rare is the outcome of interest in our data?

```r
prop.table(table(hyper$target))*100
```

```
## 
##  negative  positive 
## 95.226051  4.773949
```

The outcome appears less than 5% of the time. I've dealt with rarer events, but this is sufficiently rare for testing purposes. For testing we will run a logistic regression model on unbalanced data followed by SMOTE processing to create a more balanced dataset on which we will again run logistic regression. Both model results will be compared on the basis of the area under the curve (ROC) and the confusion matrix.


### Data splitting for modeling and cross validation of results

```r
index <- createDataPartition(hyper$target, p = .50, list = FALSE)
train <- hyper[index,]
test  <- hyper[-index,]

## SMOTE
train.smote <- SMOTE(target ~ ., train, perc.over = 100, perc.under=200)
prop.table(table(train.smote$target))*100
```

Again lets check the target rate of the train split sample and the SMOTE dervived sample.

```r
prop.table(table(train$target))*100
```

```
## 
##  negative  positive 
## 95.195954  4.804046
```

```r
prop.table(table(train.smote$target))*100
```

```
## 
## negative positive 
##       50       50
```
We have gone from a nearly 5% target rate to 37.5% targret. Indeed this is more balanced. SMOTE has used a nearest neighbor alogoritm to generate simulated observations with our target of interest - basically data has been invented. Will this perform well? Lets run logistic regression on both samples and predict on our test set.


```r
## Cross validation scheme
cvCtrl <- trainControl(method = "repeatedcv", 
                       repeats = 5, 
                       number = 10,
                       summaryFunction=twoClassSummary, 
                       classProbs=TRUE, 
                       preProc = c("center","scale","BoxCox"))

logistic <- train(target ~ .,
                  data=train,
                  method="glm", 
                  family=binomial(link="logit"),
                  metric = "ROC",
                  trControl=cvCtrl)
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
## Results
logistic.roc <- logistic$results[2]
predictions <- predict(logistic,test)
cm <- confusionMatrix(test$target, predictions)


## SMOTE logistic
logistic.smote <- train(target ~ .,
                        data=train.smote,
                        method="glm", 
                        family=binomial(link="logit"),
                        metric = "ROC",
                        trControl=cvCtrl)
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

```r
## Results
logistic.smote.roc <- logistic.smote$results[2]
predictions <- predict(logistic.smote,test)
```

```
## Warning in predict.lm(object, newdata, se.fit, scale = 1, type =
## ifelse(type == : prediction from a rank-deficient fit may be misleading
```

```r
cm.smote <- confusionMatrix(test$target, predictions)
```

### Comparing AUCs

```r
## Logistic - AUC
logistic.roc
```

```
##         ROC
## 1 0.9751412
```

```r
## SMOTE LOgistic - AUC
logistic.smote.roc
```

```
##         ROC
## 1 0.9754993
```



### Comparing confusion matrices

```r
## Logistic - Confusion Matrix
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction negative positive
##   negative     1495       11
##   positive       15       60
##                                          
##                Accuracy : 0.9836         
##                  95% CI : (0.976, 0.9892)
##     No Information Rate : 0.9551         
##     P-Value [Acc > NIR] : 3.92e-10       
##                                          
##                   Kappa : 0.8133         
##  Mcnemar's Test P-Value : 0.5563         
##                                          
##             Sensitivity : 0.9901         
##             Specificity : 0.8451         
##          Pos Pred Value : 0.9927         
##          Neg Pred Value : 0.8000         
##              Prevalence : 0.9551         
##          Detection Rate : 0.9456         
##    Detection Prevalence : 0.9526         
##       Balanced Accuracy : 0.9176         
##                                          
##        'Positive' Class : negative       
## 
```

```r
## SMOTE Logistic - Confusion Matrix
cm.smote
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction negative positive
##   negative     1453       53
##   positive        9       66
##                                         
##                Accuracy : 0.9608        
##                  95% CI : (0.95, 0.9698)
##     No Information Rate : 0.9247        
##     P-Value [Acc > NIR] : 2.150e-09     
##                                         
##                   Kappa : 0.6607        
##  Mcnemar's Test P-Value : 4.734e-08     
##                                         
##             Sensitivity : 0.9938        
##             Specificity : 0.5546        
##          Pos Pred Value : 0.9648        
##          Neg Pred Value : 0.8800        
##              Prevalence : 0.9247        
##          Detection Rate : 0.9190        
##    Detection Prevalence : 0.9526        
##       Balanced Accuracy : 0.7742        
##                                         
##        'Positive' Class : negative      
## 
```




