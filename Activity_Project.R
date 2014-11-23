#Practical Machine Learning
#Course Project

training <- read.csv("pml-training.csv", na.strings=c("", "NA"))
testing <- read.csv("pml-testing.csv", na.strings=c("", "NA"))

#removing the first index column from the data frame
training$X <- NULL

#in the process of anonymize the data with irrelevant dimension like the user name and the timestamp
remove_cols <- c("user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp")
for(col in remove_cols) {
                training[, col] <- NULL
}

#there is a lot of missing data within the variables. We can remove the dimension from the train and test dataset with too many missing values.
missing <- apply(training, 2, function(x) {sum(is.na(x))})
training <- training[, which(missing == 0)]

#based on the activity data, the data appear to have value close to zero which will be remove using the near zero variance with the caret package
library(caret)
nsv <- nearZeroVar(training)
training <- training[-nsv]
testing <- testing[-nsv]

#after cleaning the dataset, the final set of predictors used to build the classification algorithm are:
names(training)

#in building the predictive model, the classification method of random forest will be used to predict the activity class. In order to measure the model accuracy, a 10-fold K cross validation with 80:20 split will be implemented. The 80% of the data will be used for training while the remaining 20% will be used to test the model.
library(randomForest)
set.seed(1234)
obs <- c()
preds <- c()
for(i in 1:10) {
        inTrain = sample(1:dim(training)[1], size=dim(training)[1] * 0.8, replace=F)
        train_cross = training[inTrain,]
        test_cross = training[-inTrain,]
        rf <- randomForest(classe ~., data=train_cross)
        obs <- c(obs, test_cross$classe)
        preds <- c(preds, predict(rf, test_cross))
}

#confusion matrix was created to measure the accuracy of the random forest classification method
conf_mat <- confusionMatrix(table(preds, obs))
conf_mat$table

#the random forest model appears to provide a good classification for the 5 classes of actions. the accuracy is r conf_mat$overall[[1]] * 100% and it only missed classified few cases. Then we can use the training model to predict the whole dataset given the activity measurements.
final_model <- randomForest(classe~., data=training)