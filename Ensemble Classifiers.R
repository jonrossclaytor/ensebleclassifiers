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


# load the raw data
voices = read.csv("C:\\Users\\johnr\\OneDrive\\Ross\\CSC529\\Case Study 1\\voice.csv",header=TRUE, sep=",", quote='"') 
# create a scaled (mean 0, sd 1) version of the data for distance measures
scaled.voices <- voices
scaled.voices$label <- NULL # omit the original (char) lable
scaled.voices <- scale(scaled.voices)
scaled.voices <- as.data.frame(scaled.voices)
scaled.voices <- cbind(scaled.voices, label=voices$label) # add the label back to the scaled dataframe

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

# create a simple 80-20 split
set.seed(1234)
ind <- sample(2, nrow(voices), replace=TRUE, prob=c(0.8, 0.2))
train <- voices[ind==1,]
test <- voices[ind==2,]

scaled.train <- scaled.voices[ind==1,]
scaled.test <- scaled.voices[ind==2,]

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
pval # lower tail p???value 

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

# CUSTOM ENSEMBLE APPROACH

# create a matrix with the predictions from each independent categorization model
x <- cbind(pred.cv.tree, pred.cv.nb, pred.cv.knn)

# create a one-deminsional matrix for the target attribute
y <- voices$label
y <- cbind(y,y) # bind a column to make numerical
y <- y[,-2] # omit the extra column
 
# solve for coefficients with matrix multiplication
weights <- solve(t(x)%*%x)%*%t(x)%*%y

ensemble_prediction <- round(x %*% weights,0)
t <- table(y,ensemble_prediction)
confusionMatrix(t)

  # does KNN outperform the custom enseble approach?
  # null hypothesis: P > 0.9618
  # alternate hypothesis: P <= 0.9618
xbar <- 0.9842 # sample mean
mu0 <- 0.9842 # hypothesized value
sigma <- sqrt(mu0*(1-mu0)/n) # population standard deviation
z = (xbar - mu0)/(sigma/sqrt(n)) 
pval = pnorm(z)
pval # lower tail p???value 

# confirm which instance was miscategorized by each underlying approach
rownums <- c(1:3168)
x <- cbind(x,y,rownums)
missed_tree <- x[x[,1]!=x[,4],5]
missed_nb <- x[x[,2]!=x[,4],5]
missed_knn <- x[x[,3]!=x[,4],5]

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
grid.newpage()
draw.triple.venn(area1 = missed_tree, area2 = missed_knn, area3 = missed_nb, n12 = missed_knn_tree, n23 = missed_knn_nb, n13 = missed_tree_nb, 
                 n123 = missed_all, category = c("Missed Decision Tree", "Missed KNN", "Missed Naive Bayes"), lty = "blank", 
                 fill = c("skyblue", "pink1", "mediumorchid"))
