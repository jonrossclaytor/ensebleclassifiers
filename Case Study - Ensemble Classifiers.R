# install / load libraries
#install.packages("rpart")
library(rpart) # decision trees
#install.packages("caret")
library(caret) # evaluation
#install.packages("e1071")
library(e1071) # naive bayes
#install.packages("class") 
library(class) # nearest neighbors
#install.packages("klaR")
library(klaR) # cross-fold validation
#install.packages("randomForest")
library(randomForest) # random forest
#install.packages("ada")
library(ada) # adaboost
#install.packages("corrgram")
library(corrgram) # adaboost # for visualizing correlations
#install.packages('VennDiagram')
library(VennDiagram) # Venn Diagram 
#install.packages('ggplot2')
#library(ggplot2) 
#install.packages('plotly')
#library(plotly) 


# load the raw data
voices = read.csv("C:\\Users\\johnr\\OneDrive\\Ross\\CSC529\\Case Study 1\\voice.csv",header=TRUE, sep=",", quote='"') 
# create a scaled (mean 0, sd 1) version of the data for distance measures
scaled.voices <- voices
scaled.voices$label <- NULL # omit the original (char) lable
scaled.voices <- scale(scaled.voices)
scaled.voices <- as.data.frame(scaled.voices)
scaled.voices <- cbind(scaled.voices, label=voices$label) # add the label back to the scaled dataframe

# create a dataset with a binary target for use in logistic regression
bin.voices <- voices
bin.voices$binlabel[bin.voices$label=='male'] <- 1
bin.voices$binlabel[bin.voices$label!='male'] <- 0
bin.voices$label <- NULL

# create specific subsets of the data by measurement for correlation analysis
  # frequency
voices.freq <- voices[1:6]
voices.freq <- cbind(voices.freq,voices[11])
voices.freq <- cbind(voices.freq,voices[12])
cor(voices.freq)
corrgram(voices.freq)
  # fundamental frequency
voices.fund.freq <- voices[13:15]
cor(voices.fund.freq)
corrgram(voices.fund.freq)
  # dominant frequency
voices.dom.freq <- voices[16:19]
cor(voices.dom.freq)
corrgram(voices.dom.freq)
  # means of all three frequencies
voices.mean.freq <- voices[1]
voices.mean.freq <- cbind(voices.mean.freq,voices[13])
voices.mean.freq <- cbind(voices.mean.freq,voices[16])
cor(voices.mean.freq)
corrgram(voices.mean.freq)

# create subsets of the scaled data
scaled.voices.freq <- scaled.voices[1:6]
scaled.voices.freq <- cbind(scaled.voices.freq,scaled.voices[11])
scaled.voices.freq <- cbind(scaled.voices.freq,scaled.voices[12])
# fundamental frequency
scaled.voices.fund.freq <- scaled.voices[13:15]
# dominant frequency
scaled.voices.dom.freq <- scaled.voices[16:19]



# create a simple 80-20 split
set.seed(1234)
ind <- sample(2, nrow(voices), replace=TRUE, prob=c(0.8, 0.2))

train <- voices[ind==1,]
test <- voices[ind==2,]
bin.train <- bin.voices[ind==1,]
bin.test <- bin.voices[ind==2,]
scaled.train <- scaled.voices[ind==1,]
scaled.test <- scaled.voices[ind==2,]

# build a linear discriminant analysis model
model.lda <- lda(label ~ . , data=train)
summary(model.lda)
predict.lda <- predict(model.lda,test)
# create a confusion matrix
t <- table(predict.lda$class,test$label)
confusionMatrix(t)


# conduct 10-fold cross validation
set.seed(1234)
train_control <- trainControl(method="cv", number=10)
model <- train(label ~ . , data=voices, trControl=train_control, method="lda")
pred.cv.lda <- predict(model,voices)
t <- table(pred.cv.lda$class,voices$label)
confusionMatrix(t)


# use bagging to see if we can improve performance on the lda model
options(warn=-1) # temporarily suppress warning -- this model will throw 'variables are collinear'
predictions<-NULL
library(foreach)
length_divisor<-4
iterations<-100
predictions<-foreach(m=1:iterations,.combine=cbind.data.frame) %do% {
  training_positions <- sample(nrow(train), size=floor((nrow(train)/length_divisor)))
  train_pos<-1:nrow(train) %in% training_positions
  model <- lda(label ~ . , data=train[train_pos,])
  predict<-predict(model,newdata=test)
  predict$class
}
predictions<-apply(predictions,1,function(x) names(which.max(table(x))))
t <- table(predictions,test$label)
confusionMatrix(t)
options(warn=0)

# use hypthesis testing to confirm whether bagging improved our LDA model

n <- dim(test)[1]
# does the LDA model without bagging have higher accuracy than the model with bagging?
# null hypothesis: P > 0.9702 
# alternate hypothesis: P <= 0.9702
xbar <- 0.9686 # sample mean
mu0 <- 0.9702 # hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n)
z = (xbar - mu0)/(sigma/sqrt(n)) # population standard deviation
pval = pnorm(z)
pval # lower tail 
# Reject null hypothesis

lda_delta <- (0.9702 - 0.9686) / 0.9686

# build a logistic regression model using glm
model.lr <- glm(binlabel ~.,family=binomial(link='logit'),data=bin.train)
summary(model.lr)
pred.test.lr <- predict(model.lr, bin.test, type = "response")
pred.test.lr <- ifelse(pred.test.lr > 0.5,1,0)

# create a confusion matrix
t <- table(pred.test.lr,bin.test$binlabel)
confusionMatrix(t)


# conduct 10-fold cross validation
set.seed(1234)
train_control <- trainControl(method="cv", number=10)
model <- train(binlabel ~ . , data=bin.voices, trControl=train_control, method="glm")
pred.cv.log <- predict(model,bin.voices)
pred.cv.log <- ifelse(pred.cv.log > 0.5,1,0)


t <- table(pred.cv.log,bin.voices$binlabel)
confusionMatrix(t)


# convert the predictions back to male and female
new.pred.cv.log <- NA
m <- pred.cv.log == 1
new.pred.cv.log[m] <- "male"
new.pred.cv.log[!m] <- "female"
pred.cv.log <- new.pred.cv.log
pred.cv.log <- as.factor(pred.cv.log)
new.pred.cv.log <- NULL

t <- table(pred.cv.log,voices$label)
confusionMatrix(t)


# use bagging to see if we can improve performance on the logistic regression model
options(warn=-1) # temporarily suppress warning -- this model will throw 'prediction from a rank-deficient fit may be misleading' due to the colinearity in the data
predictions<-NULL
library(foreach)
length_divisor<-4
iterations<-1000
predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
  training_positions <- sample(nrow(bin.train), size=floor((nrow(bin.train)/length_divisor)))
  train_pos<-1:nrow(bin.train) %in% training_positions
  model <- glm(binlabel ~.,family=binomial(link='logit'),data=bin.train[train_pos,])
  pred.model <- predict(model,newdata=bin.test, type = "response")
  ifelse(pred.model > 0.5,1,0)
}
predictions<-apply(predictions,1,function(x) names(which.max(table(x))))
t <- table(predictions,bin.test$binlabel)
confusionMatrix(t)
options(warn=0) # turn back on warnings


n <- dim(test)[1]
# does the logistic regresion model without bagging have higher accuracy than the model with bagging?
# null hypothesis: P > 0.9765 
# alternate hypothesis: P <= 0.9765
xbar <- 0.9765 # sample mean
mu0 <- 0.9765 # hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n)
z = (xbar - mu0)/(sigma/sqrt(n)) # population standard deviation
pval = pnorm(z)
pval # lower tail 
# Fail to reject null hypothesis

log_delta <- 0

# build a decision tree using the rpart package
model.tree <- rpart(label ~ . , data=voices)
summary(model.tree)
plot(model.tree)
text(model.tree, pretty=0)

# use the model to make predictions
pred.test.tree <- predict(model.tree, test, type = "class")

# create a confusion matrix
t <- table(pred.test.tree,test$label)
confusionMatrix(t)

# conduct 10-fold cross validation
set.seed(1234)
train_control <- trainControl(method="cv", number=10)
model <- train(label ~ . , data=voices, trControl=train_control, method="rpart")
pred.cv.tree <- predict(model,voices)
t <- table(pred.cv.tree,voices$label)
confusionMatrix(t)

# use bagging to see if we can improve performance on the decision tree model
predictions<-NULL
library(foreach)
length_divisor<-4
iterations<-1000
predictions<-foreach(m=1:iterations,.combine=cbind.data.frame) %do% {
  training_positions <- sample(nrow(train), size=floor((nrow(train)/length_divisor)))
  train_pos<-1:nrow(train) %in% training_positions
  model <- rpart(label ~ . , data=train[train_pos,])
  predict(model,newdata=test,type = "class")
}
predictions<-apply(predictions,1,function(x) names(which.max(table(x))))
t <- table(predictions,test$label)
confusionMatrix(t)

n <- dim(test)[1]
# does the decision tree model without bagging have higher accuracy than the model with bagging?
# null hypothesis: P > 0.9717 
# alternate hypothesis: P <= 0.9618
xbar <- 0.9618 # sample mean
mu0 <- 0.9717 # hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n)
z = (xbar - mu0)/(sigma/sqrt(n)) # population standard deviation
pval = pnorm(z)
pval # lower tail 
# Reject null hypothesis

dt_delta <- (0.9717 - 0.9618) / 0.9618

# does base decision tree with bagging improve upon accuracy of base knn model?
# null hypothesis: P > 0.9842 
# alternate hypothesis: P <= 0.9842
n <- dim(test)[1]
xbar <- 0.9717 # sample mean
mu0 <-  0.9842# hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n)
z = (xbar - mu0)/(sigma/sqrt(n)) # population standard deviation
pval = pnorm(z)
pval # lower tail 
# Reject null hypothesis

# NAIVE BAYES
# create a model 
model.nb <- naiveBayes(label ~ . , data=voices)

# compute the misclassification rate for the test data
pred.test.nb <- predict(model.nb, test, type = "class")
mean(pred.test.nb != test$label)

t <- table(pred.test.nb,test$label)
confusionMatrix(t)

# conduct 10-fold cross validation
set.seed(1234)
train_control <- trainControl(method="cv", number=10)
model <- train(label ~ . , data=voices, trControl=train_control, method="nb")
pred.cv.nb <- predict(model,voices)
t <- table(pred.cv.nb,voices$label)
confusionMatrix(t)

# use bagging to see if we can improve performance on the naive bayes model
predictions<-NULL
library(foreach)
length_divisor<-4
iterations<-1000
predictions<-foreach(m=1:iterations,.combine=cbind.data.frame) %do% {
  training_positions <- sample(nrow(train), size=floor((nrow(train)/length_divisor)))
  train_pos<-1:nrow(train) %in% training_positions
  model <- naiveBayes(label ~ . , data=train[train_pos,])  
  predict(model,newdata=test,type = "class")
}
predictions<-apply(predictions,1,function(x) names(which.max(table(x))))
t <- table(predictions,test$label)
confusionMatrix(t)

nb_delta <- (0.898 - 0.9299) / 0.9299


n <- dim(test)[1]
# does the naive bayes model WITH bagging have higher accuracy than the model WITHOUT bagging?
# null hypothesis: P > 0.9299
# alternate hypothesis: P <= 0.9299
xbar <- 0.898 # sample mean
mu0 <-  0.9299# hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n)
z = (xbar - mu0)/(sigma/sqrt(n)) # population standard deviation
pval = pnorm(z)
pval # lower tail 
# Reject null hypothesis


# NEAREST NEIGHBORS
set.seed(1234)
pred.test.knn <- knn(scaled.train[,1:20], scaled.test[,1:20], scaled.train[,21], k = 3)
mean(pred.test.knn != test$label) 

t <- table(pred.test.knn,test$label)
confusionMatrix(t)

# conduct 10-fold cross validation
set.seed(1234)
train_control <- trainControl(method="cv", number=10)
knnFit <- train(label ~ . , data=scaled.voices, method = "knn", trControl = train_control, preProcess = c("center","scale"), tuneLength = 20)

knnFit

# now we can build a model with the optimal number of nearest neighbors (5)
set.seed(1234)
pred.test.knn <- knn(scaled.train[,1:20], scaled.test[,1:20], scaled.train[,21], k = 5)
t <- table(pred.test.knn,test$label)
confusionMatrix(t)

# complete cross-fold validation using "train" and the "knn" method
set.seed(1234)
train_control <- trainControl(method="cv", number=10)
model <- train(label ~ . , data=scaled.voices, trControl=train_control, method="knn")
pred.cv.knn <- predict(model,scaled.voices)
t <- table(pred.cv.knn,scaled.voices$label)
confusionMatrix(t)


# use bagging to see if we can improve performance on the knn model
predictions<-NULL
library(foreach)
length_divisor<-4
iterations<-1000
predictions<-foreach(m=1:iterations,.combine=cbind.data.frame) %do% {
  training_positions <- sample(nrow(train), size=floor((nrow(train)/length_divisor)))
  train_pos<-1:nrow(train) %in% training_positions
  knn(scaled.train[train_pos,1:20], scaled.test[,1:20], scaled.train[train_pos,21], k = 5)
}
predictions<-apply(predictions,1,function(x) names(which.max(table(x))))
t <- table(predictions,test$label)
confusionMatrix(t)


n <- dim(test)[1]
# does the knn model WITH bagging have higher accuracy than the model WITHOUT bagging?
# null hypothesis: P > 0.9299
# alternate hypothesis: P <= 0.9299
xbar <- 0.9655 # sample mean
mu0 <-  0.9842 # hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n)
z = (xbar - mu0)/(sigma/sqrt(n)) # population standard deviation
pval = pnorm(z)
pval # lower tail 
# Reject null hypothesis

knn_delta <- (0.9655 - 0.9842) / 0.9842

# generate a bar chart to show the results of bagging
delta <- as.table(cbind(lda_delta*100,log_delta*100,dt_delta*100,nb_delta*100,knn_delta*100))
barplot(delta, ylab="% Change in Accuracy", ylim=c(-5,5),col=c("darkblue"),names.arg=c("lda","logistic reg","decition tree","naive bayes","knn"))


# perform hypothesis testing to compare independent models
n <- 3168 # from teh cross validation
  # does Naive Bayes perform better than Decision Tree?
  # null hypothesis: P > 0.9299 
  # alternate hypothesis: P <= 0.9299
xbar <- 0.9299 # sample mean
mu0 <- 0.9618 # hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n)
z = (xbar - mu0)/(sigma/sqrt(n)) # population standard deviation
pval = pnorm(z)
pval # lower tail 

  # does decision tree perform better than K Neareast Neighbor??
  # null hypothesis: P > 0.9618
  # alternate hypothesis: P <= 0.9618
xbar <- 0.9618 # sample mean
mu0 <- 0.9842 # hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n) # population standard deviation
z = (xbar - mu0)/(sigma/sqrt(n)) 
pval = pnorm(z)
pval # lower tail p???value 

# RANDOM FOREST
# use random forest to build 500 decision trees
rf.500 <- randomForest(label ~ ., data=voices, ntree=500)
pred.rf.500 <- predict(rf.500, voices, type = "class")
t <- table(pred.rf.500,voices$label)
confusionMatrix(t)

# ADABOOST
model.ada <- ada(voices[,1:20],voices[,21],iter=500)
pred.ada <- predict(model.ada,voices)
t <- table(pred.ada,voices$label)
confusionMatrix(t)

# STACKING

# create a matrix with the predictions from each independent categorization model
x <- cbind(pred.cv.tree, pred.cv.nb, pred.cv.knn, pred.cv.lda, pred.cv.log)

# create a one-deminsional matrix for the target attribute
y <- voices$label
y <- cbind(y,y) # bind a column to make numerical
y <- y[,-2] # omit the extra column

# allow the algorithms to categorize using majority voting
predictions<-apply(x,1,function(x) names(which.max(table(x))))
t <- table(predictions,y)
confusionMatrix(t)


# attempt to solve for coefficients with matrix multiplication # error: system is computationally singular
weights <- solve(t(x)%*%x)%*%t(x)%*%y

# use linear regression to estimate the weights
# add the target attribute back to the data
x.y <- cbind(x,y)
x.y <- as.data.frame(x.y)
lm.model <- lm(y ~.,data=x.y)
summary(lm.model)

lm.predict <- round(predict(lm.model,x.y),0)
t <- table(lm.predict ,y)
confusionMatrix(t)


# perform stacking based on the different subsets of variables
voices.freq <- cbind.data.frame(voices.freq,label=voices$label)
voices.fund.freq <- cbind.data.frame(voices.fund.freq,label=voices$label)
voices.dom.freq <- cbind.data.frame(voices.dom.freq,label=voices$label)

scaled.voices.freq <- cbind.data.frame(scaled.voices.freq,label=voices$label)
scaled.voices.fund.freq <- cbind.data.frame(scaled.voices.fund.freq,label=voices$label)
scaled.voices.dom.freq <- cbind.data.frame(scaled.voices.dom.freq,label=voices$label)


# build decision trees on each underlying frequency type using cross validation
set.seed(1234)
train_control <- trainControl(method="cv", number=10)

# DECISION TREES
  # freq
model <- train(label ~ . , data=voices.freq, trControl=train_control, method="rpart")
pred.freq.tree <- predict(model,voices)
t <- table(pred.freq.tree,voices$label)
confusionMatrix(t)
  # fund
model <- train(label ~ . , data=voices.fund.freq, trControl=train_control, method="rpart")
pred.fund.tree <- predict(model,voices)
t <- table(pred.fund.tree,voices$label)
confusionMatrix(t)
  # dom
model <- train(label ~ . , data=voices.dom.freq, trControl=train_control, method="rpart")
pred.dom.tree <- predict(model,voices)
t <- table(pred.dom.tree,voices$label)
confusionMatrix(t)

decision.stack <- cbind.data.frame(pred.freq.tree,pred.fund.tree,pred.dom.tree)
pred.decision.stack <- apply(decision.stack,1,function(x) names(which.max(table(x))))
t <- table(pred.decision.stack,voices$label)
confusionMatrix(t)

# KNN
  # freq
model <- train(label ~ . , data=scaled.voices.freq, trControl=train_control, method="knn")
pred.freq.knn <- predict(model,scaled.voices)
t <- table(pred.freq.knn,voices$label)
confusionMatrix(t)
  # fund
model <- train(label ~ . , data=scaled.voices.fund.freq, trControl=train_control, method="knn")
pred.fund.knn <- predict(model,scaled.voices)
t <- table(pred.fund.knn,voices$label)
confusionMatrix(t)
  # dom
model <- train(label ~ . , data=scaled.voices.dom.freq, trControl=train_control, method="knn")
pred.dom.knn <- predict(model,scaled.voices)
t <- table(pred.dom.knn,voices$label)
confusionMatrix(t)

knn.stack <- cbind.data.frame(pred.freq.knn,pred.fund.knn,pred.dom.knn)
pred.knn.stack <- apply(knn.stack,1,function(x) names(which.max(table(x))))
t <- table(pred.knn.stack,voices$label)
confusionMatrix(t)


# comibe the results of both 
decision.knn.stack <- cbind.data.frame(pred.freq.tree,pred.fund.tree,pred.dom.tree,pred.freq.knn,pred.fund.knn,pred.dom.knn)
pred.decision.knn.stack <- apply(decision.knn.stack,1,function(x) names(which.max(table(x))))
t <- table(pred.decision.knn.stack,voices$label)
confusionMatrix(t)


# NOTE - THIS SECTION IS NO LONGER INCLUDED IN THE REPORT

# confirm which instance was miscategorized by each underlying approach
rownums <- c(1:3168)
x <- cbind(x,y,rownums)
missed_tree <- x[x[,1]!=x[,6],7]
missed_nb <- x[x[,2]!=x[,6],7]
missed_knn <- x[x[,3]!=x[,6],7]
missed_lda <- x[x[,4]!=x[,6],7]
missed_log <- x[x[,5]!=x[,6],7]

# determine the counts of instances that were missed
missed_all <- Reduce(intersect, list(missed_knn,missed_tree,missed_nb))
missed_knn_nb <- Reduce(intersect, list(missed_knn,missed_nb))
missed_knn_tree <- Reduce(intersect, list(missed_knn,missed_tree))
missed_tree_nb <- Reduce(intersect, list(missed_tree,missed_nb))

missed_all <- length(missed_all)
missed_knn_nb <- length(missed_knn_nb)
missed_knn_tree <- length(missed_knn_tree)
missed_tree_nb <- length(missed_tree_nb)
missed_tree <- length(missed_tree)
missed_knn <- length(missed_knn)
missed_nb <- length(missed_nb)

# create a Venn Diagram with the results
#grid.newpage()
#draw.triple.venn(area1 = missed_tree, area2 = missed_knn, area3 = missed_nb, n12 = missed_knn_tree, n23 = missed_knn_nb, n13 = missed_tree_nb, 
#                 n123 = missed_all, category = c("Missed Decision Tree", "Missed KNN", "Missed Naive Bayes"), lty = "blank", 
#                 fill = c("skyblue", "pink1", "mediumorchid"))
