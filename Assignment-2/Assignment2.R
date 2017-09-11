#install.packages("nnet", dependencies = TRUE)
#install.packages("e1071", dependencies = TRUE)
#install.packages("rpart", dependencies = TRUE)


library("nnet")
library("e1071")
library("rpart")



arg <- commandArgs(TRUE)
#dataURL<-as.character(arg[1])
datasetURL <- as.character("http://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data")
header <- as.logical(F)
d <- read.csv(datasetURL,header= header)
#c_id <- as.integer(args[2])
classid <- as.integer(9)

Class<-d[,as.integer(classid)]  

for(i in 1:2)
{
  
  
  sampleInstances<-sample(1:nrow(d),size = 0.9*nrow(d))
  trainingData<-d[sampleInstances,]
  testData<-d[-sampleInstances,]
  
  cat("\n\nRunning sample ",i,":\n\n")
  
  set.seed(123)
  
  d[,classid] <- as.numeric(d[,classid])
  
  
  trainingData[,classid] <- as.factor(trainingData[,classid])
  testData[,classid] <- as.factor(testData[,classid])
  
  
  ClassName<- names(trainingData[classid])
  c_names <- colnames(d)
  formula1 <- as.formula(paste(ClassName,"~."))
  ######################################
  
  model <- naiveBayes(as.formula(formula1), data = trainingData,na.action = na.pass)
  pred <- predict(model, testData, type = "class")
  
  method="Naive Bayesian"
  accuracy <- sum(testData[,classid]==pred)/length(pred)
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  ######################################
  
  model <- rpart (as.formula(formula1), data=trainingData,method="class")
  pred <- predict(model, testData, decision.values = TRUE, type = "class")
  
  method="Decision Trees"
  accuracy <- sum(testData[,classid]==pred)/length(pred)
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  #####################################
  
  
  model <- svm(as.formula(formula1), data = trainingData, na.action = na.pass, kernel="linear")
  pred <- predict(model, testData, type = "class")
  
  method="SVM"
  accuracy <- sum(testData[,classid]==pred)/length(pred)
  cat("Method = ", method,", accuracy= ", accuracy,"\n")
  
  ##################################### 
  
  neural <- nnet(formula1, trainingData,size=4,maxit=1000,decay=0.001,trace = FALSE)
  predn <- predict(neural,testData,type="class")
  accun <- mean(predn == testData[ ,classid])
  method="Neural Network" 
  cat("Method = ", method,", Accuracy = ", accun,"\n")
  
  ##################################### 
  
  neural <- nnet(formula1, trainingData,size=0,maxit=1000,skip=TRUE,decay=0.001,trace = FALSE)
  predn <- predict(neural,testData,type="class")
  accun <- mean(predn == testData[ ,classid])
  method="Perceptron" 
  cat("Method = ", method,", Accuracy = ", accun,"\n")
  
}
