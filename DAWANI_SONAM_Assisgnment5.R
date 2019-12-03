library(caret)
library(gbm)
library(randomForest)
library(mlbench)
library(e1071)
library('RANN')
data(scat)
str(scat)

########## 1. Set the Species column as the target/outcome and convert it to numeric.
########## Answer1
#Converting 'Species' column to numeric
outcomeName<-'Species'
scat$Species<-ifelse(scat$Species=='gray_fox',3,ifelse(scat$Species=='coyote',2,1))
str(scat)
View(scat)

########## 2. Remove the Month, Year, Site, Location features.
########## Answer2
#Dropping specified columns
drops <- c("Month","Year","Site","Location")
scat_processed<-scat[ , !(names(scat) %in% drops)]
View(scat_processed)

#As we dont have categorical value, we dont have to create dummy variables, and we can proceed to convert back our target feature to caterogical value
#Convering 'Species' column back to categorical
scat_processed$Species<-as.factor(scat_processed$Species)
str(scat_processed)

########## 3. Check if any values are null. If there are, impute missing values using KNN.
########## Answer3
sum(is.na(scat_processed))
preProcValues <- preProcess(scat_processed, method = c("knnImpute","center","scale"))
scat_processed <- predict(preProcValues, scat_processed)
sum(is.na(scat_processed))

########## 4. Converting every categorical variable to numerical (if needed). 
########## Answer4
#Not Needed as we dont have any categorical value feature (in predictors)
str(scat_processed)

########## 5. With a seed of 100, 75% training, 25% testing. Build the following models: randomforest, neural net, naive bayes and GBM.
########## Answer5
#Spliting training set into two parts based on outcome: 75% and 25%
set.seed(100)
index <- createDataPartition(scat_processed$Species, p=0.75, list=FALSE)
trainSet <- scat_processed[ index,]
testSet <- scat_processed[-index,]
str(trainSet)
str(testSet)

outcomeName<-'Species'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]

#Building Models

model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')

#model_rf
# a) Randomforest - model summarization
print(model_rf)

# b) Randomforest - plot of variable of importance
varImp(object=model_rf)
plot(varImp(object=model_rf),main="RF - Variable Importance")

# c) Randomforest - confusion matrix
predictions<-predict.train(object=model_rf,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])

#model_nnet
# a) Neuralnet - model summarization
print(model_nnet)

# b) Neuralnet - plot of variable of importance
varimpnnetDF<-varImp(object=model_nnet)
varimpnnetDF_imp<-(varimpnnetDF$importance)
varimpnnetDF_impDF<-data.frame(varimpnnetDF_imp)
varimpnnetDF_impDF<-data.frame(varimpnnetDF_impDF$Overall)
row.names(varimpnnetDF_impDF)<-row.names(varimpnnetDF_imp)
varimpnnetDF_impDF$Variable<-row.names(varimpnnetDF_imp)
names(varimpnnetDF_impDF)<-c("Importance","Variables")
barplot(varimpnnetDF_impDF$Importance, names = varimpnnetDF_impDF$Variables,las=2)

# c) Neuralnet - confusion matrix
predictions<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])

#model_nb
# a) Naivebayes - model summarization
print(model_nb)

# b) Naivebayes - plot of variable of importance
varImp(object=model_nb)
plot(varImp(object=model_nb),main="NB - Variable Importance")

# c) Naivebayes - confusion matrix
predictions<-predict.train(object=model_nb,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])

#model_gbm
# a) GBM - model summarization
print(model_gbm)

# b) GBM - plot of variable of importance
varImp(object=model_gbm)
plot(varImp(object=model_gbm),main="GBM - Variable Importance")

# c) GBM - confusion matrix
predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
confusionMatrix(predictions,testSet[,outcomeName])

########## 6. For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) create
#             and display a data frame that has the following columns: ExperimentName, accuracy, kappa.
#             Sort the data frame by accuracy.
########## Answer6

#Finding best model for Randomforest
model_rf_results_df=model_rf$results
model_rf_results_ordered_df=model_rf$results[order(-model_rf_results_df$Accuracy),]
model_rf_results_ordered_AK_df <- data.frame(model_rf_results_ordered_df$Accuracy,model_rf_results_ordered_df$Kappa)
model_rf_results_ordered_AK_df_r1 <- data.frame("model_rf",model_rf_results_ordered_AK_df[1,])
names(model_rf_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

#Finding best model for Neuralnet
model_nnet_results_df=model_nnet$results
model_nnet_results_ordered_df=model_nnet$results[order(-model_nnet_results_df$Accuracy),]
model_nnet_results_ordered_AK_df <- data.frame(model_nnet_results_ordered_df$Accuracy,model_nnet_results_ordered_df$Kappa)
model_nnet_results_ordered_AK_df_r1 <- data.frame("model_nnet",model_nnet_results_ordered_AK_df[1,])
names(model_nnet_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

#Finding best model for Naivebayes
model_nb_results_df=model_nb$results
model_nb_results_ordered_df=model_nb$results[order(-model_nb_results_df$Accuracy),]
model_nb_results_ordered_AK_df <- data.frame(model_nb_results_ordered_df$Accuracy,model_nb_results_ordered_df$Kappa)
model_nb_results_ordered_AK_df_r1 <- data.frame("model_nb",model_nb_results_ordered_AK_df[1,])
names(model_nb_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

#Finding best model for GradientBoosting
model_gbm_results_df=model_gbm$results
model_gbm_results_ordered_df=model_gbm$results[order(-model_gbm_results_df$Accuracy),]
model_gbm_results_ordered_AK_df <- data.frame(model_gbm_results_ordered_df$Accuracy,model_gbm_results_ordered_df$Kappa)
model_gbm_results_ordered_AK_df_r1 <- data.frame("model_gbm",model_gbm_results_ordered_AK_df[1,])
names(model_gbm_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

#Integrading best models
summarydf <- rbind(model_gbm_results_ordered_AK_df_r1, model_rf_results_ordered_AK_df_r1,model_nnet_results_ordered_AK_df_r1,model_nb_results_ordered_AK_df_r1)
summarydf=summarydf[order(-summarydf$Accuracy),]
rownames(summarydf) <- NULL
print(summarydf)

########## 7. Tune the GBM model using tune length = 20 and: a) print the model summary and b) plot the models.
########## Answer7
#Tuning GBM - using tune length 20
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
model_gbm_tl<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
print(model_gbm_tl)
plot(model_gbm_tl)

#Finding best model for GradientBoosting
model_gbm_tl_results_df=model_gbm_tl$results
model_gbm_tl_results_ordered_df=model_gbm_tl$results[order(-model_gbm_tl_results_df$Accuracy),]
model_gbm_tl_results_ordered_AK_df <- data.frame(model_gbm_tl_results_ordered_df$Accuracy,model_gbm_tl_results_ordered_df$Kappa)
model_gbm_tl_results_ordered_AK_df_r1 <- data.frame("model_gbm_tuneLength20",model_gbm_tl_results_ordered_AK_df[1,])
names(model_gbm_tl_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

summarydf <- rbind(model_gbm_results_ordered_AK_df_r1, model_rf_results_ordered_AK_df_r1,model_nnet_results_ordered_AK_df_r1,model_nb_results_ordered_AK_df_r1,model_gbm_tl_results_ordered_AK_df_r1)
summarydf=summarydf[order(-summarydf$Accuracy),]
rownames(summarydf) <- NULL
print(summarydf)

########## 8. Using GGplot and gridExtra to plot all variable of importance plots into one single plot.
########## Answer8
#all variable of importance plots into one single plot
RF_gg<-ggplot(varImp(object=model_rf))+ggtitle("RF - Variable Importance")
GBM_gg<-ggplot(varImp(object=model_gbm))+ggtitle("GBM - Variable Importance")
NNET_gg<-ggplot(data=varimpnnetDF_impDF, aes(x=Variables, y=Importance)) + ggtitle("NNET - Variable Importance")+
  geom_bar(stat="identity") + coord_flip()
NB_gg<-ggplot(varImp(object=model_nb))+ggtitle("NB - Variable Importance")
grid.arrange(RF_gg,NNET_gg,NB_gg,GBM_gg, ncol= 2 )


########## Answer8 Alternate way
#Plot importance variable
vimp_gbm<-varImp(object=model_gbm)
gbm_DF<-vimp_gbm$importance
gbm_DF$Variable <- rownames(gbm_DF)
gbm_DF$Model <- c("GBM")
names(gbm_DF)<-c("Importance","Variable","Model")

gbm_DF_plot<-ggplot(data=gbm_DF, aes(x=Variable, y=Importance, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme(axis.text.x = element_text(angle = 45))


vimp_rf<-varImp(object=model_rf)
rf_DF<-vimp_rf$importance
rf_DF$Variable <- rownames(rf_DF)
rf_DF$Model <- c("RF")
names(rf_DF)<-c("Importance","Variable","Model")

rf_DF_plot<-ggplot(data=rf_DF, aes(x=Variable, y=Importance, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme(axis.text.x = element_text(angle = 45))

vimp_nnet<-varImp(object=model_nnet)
nnet_DF<-vimp_nnet$importance
nnet_DF<-data.frame(nnet_DF)
nnet_DF <- data.frame(nnet_DF$Overall)
nnet_DF$Variable <- rownames(rf_DF)
nnet_DF$Model <- c("NNET")
names(nnet_DF)<-c("Importance","Variable","Model")

nnet_DF_plot<-ggplot(data=nnet_DF, aes(x=Variable, y=Importance, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme(axis.text.x = element_text(angle = 45))

vimp_nb<-varImp(object=model_nb)
nb_DF<-vimp_nb$importance
nb_bobcat_DF <- data.frame(nb_DF$X1)
nb_bobcat_DF$Variable <- rownames(rf_DF)
nb_bobcat_DF$Model <- c("NB_bobcat")
names(nb_bobcat_DF)<-c("Importance","Variable","Model")

nb_bobcat_DF_plot<-ggplot(data=nb_bobcat_DF, aes(x=Variable, y=Importance, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme(axis.text.x = element_text(angle = 45))

vimp_nb<-varImp(object=model_nb)
nb_DF<-vimp_nb$importance
nb_coyote_DF <- data.frame(nb_DF$X2)
nb_coyote_DF$Variable <- rownames(rf_DF)
nb_coyote_DF$Model <- c("NB_coyote")
names(nb_coyote_DF)<-c("Importance","Variable","Model")

nb_coyote_DF_plot<-ggplot(data=nb_coyote_DF, aes(x=Variable, y=Importance, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme(axis.text.x = element_text(angle = 45))

vimp_nb<-varImp(object=model_nb)
nb_DF<-vimp_nb$importance
nb_gray_fox_DF <- data.frame(nb_DF$X3)
nb_gray_fox_DF$Variable <- rownames(rf_DF)
nb_gray_fox_DF$Model <- c("NB_gray_fox")
names(nb_gray_fox_DF)<-c("Importance","Variable","Model")

nb_gray_fox_DF_plot<-ggplot(data=nb_gray_fox_DF, aes(x=Variable, y=Importance, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme(axis.text.x = element_text(angle = 45))

grid.arrange(gbm_DF_plot,rf_DF_plot,nnet_DF_plot,nb_bobcat_DF_plot,nb_coyote_DF_plot, nb_gray_fox_DF_plot, ncol= 2 )


#extra
varimpdf <- rbind(gbm_DF, rf_DF,nnet_DF,nb_bobcat_DF,nb_coyote_DF,nb_gray_fox_DF)
print(varimpdf)
varimpdf=varimpdf[order(-varimpdf$Importance),]

ggplot(data=varimpdf) + 
  geom_line(aes(x=Variable, y=Importance, group=Model, color=Model )) 
#extra-end

########## 9. Which model performs the best? and why do you think this is the case? Can we accurately predict species on this dataset?
########## Answer9
# With the below results:
# Model  Accuracy     Kappa
# 3             model_nnet 0.6908921 0.4788325
# 5 model_gbm_tuneLength20 0.6847222 0.4705875
# 4               model_nb 0.6624052 0.4391168
# 2               model_rf 0.6507490 0.4248715
# 1              model_gbm 0.6275335 0.3671868
# We can say that best performing model is of Neural Net. As it has the highest accuracy in comparision to other model's accuracy when taken the best parameters of for the models. 
# Hence, we can say that we can predict Species with 69% accuracy. 

########## 10. a. Using feature selection with rfe in caret and the repeatedcv method: Find the top 3 predictors and build the same models as in 6 and 8 with the same parameters
########## Answer10-a
#RFE feature selection
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Species'
Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],rfeControl = control)
optVariables=Loan_Pred_Profile$optVariables
predictors<-head(optVariables,n=3)
print(predictors)

##Repeating 6 to 8

##### Repeating Answer6 with Feature Selection
#Creating Models with Feature Selection
fs_model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
fs_model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
fs_model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
fs_model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='nb')

#model_gbm
fs_model_gbm_results_df<-fs_model_gbm$results
fs_model_gbm_results_ordered_df<-fs_model_gbm$results[order(-fs_model_gbm_results_df$Accuracy),]
fs_model_gbm_results_ordered_AK_df <- data.frame(fs_model_gbm_results_ordered_df$Accuracy,fs_model_gbm_results_ordered_df$Kappa)
fs_model_gbm_results_ordered_AK_df_r1 <- data.frame("fs_model_gbm",fs_model_gbm_results_ordered_AK_df[1,])
names(fs_model_gbm_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

#model_rf
fs_model_rf_results_df=fs_model_rf$results
fs_model_rf_results_ordered_df=fs_model_rf$results[order(-fs_model_rf_results_df$Accuracy),]
fs_model_rf_results_ordered_AK_df <- data.frame(fs_model_rf_results_ordered_df$Accuracy,fs_model_rf_results_ordered_df$Kappa)
fs_model_rf_results_ordered_AK_df_r1 <- data.frame("fs_model_rf",fs_model_rf_results_ordered_AK_df[1,])
names(fs_model_rf_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

#model_nnet
fs_model_nnet_results_df=fs_model_nnet$results
fs_model_nnet_results_ordered_df=fs_model_nnet$results[order(-fs_model_nnet_results_df$Accuracy),]
fs_model_nnet_results_ordered_AK_df <- data.frame(fs_model_nnet_results_ordered_df$Accuracy,fs_model_nnet_results_ordered_df$Kappa)
fs_model_nnet_results_ordered_AK_df_r1 <- data.frame("fs_model_nnet",fs_model_nnet_results_ordered_AK_df[1,])
names(fs_model_nnet_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")

#model_nb
fs_model_nb_results_df=fs_model_nb$results
fs_model_nb_results_ordered_df=fs_model_nb$results[order(-fs_model_nb_results_df$Accuracy),]
fs_model_nb_results_ordered_AK_df <- data.frame(fs_model_nb_results_ordered_df$Accuracy,fs_model_nb_results_ordered_df$Kappa)
fs_model_nb_results_ordered_AK_df_r1 <- data.frame("fs_model_nb",fs_model_nb_results_ordered_AK_df[1,])
names(fs_model_nb_results_ordered_AK_df_r1)<-c("Model","Accuracy","Kappa")


fs_summarydf <- rbind(fs_model_gbm_results_ordered_AK_df_r1, fs_model_rf_results_ordered_AK_df_r1,fs_model_nnet_results_ordered_AK_df_r1,fs_model_nb_results_ordered_AK_df_r1)
fs_summarydf=fs_summarydf[order(-fs_summarydf$Accuracy),]
rownames(fs_summarydf)<-NULL
print(fs_summarydf)

##### Repeating Answer7 with Feature Selection
#Tuning GBM - using tune length 20
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)
model_gbm_tl<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=20)
print(model_gbm_tl)
plot(model_gbm_tl)

##### Repeating Answer8 with Feature Selection

fs_varimpnnetDF<-varImp(object=fs_model_nnet)
fs_varimpnnetDF_imp<-(fs_varimpnnetDF$importance)
fs_varimpnnetDF_impDF<-data.frame(fs_varimpnnetDF_imp)
fs_varimpnnetDF_impDF<-data.frame(fs_varimpnnetDF_impDF$Overall)
row.names(fs_varimpnnetDF_impDF)<-row.names(fs_varimpnnetDF_imp)
fs_varimpnnetDF_impDF$Variable<-row.names(fs_varimpnnetDF_imp)
names(fs_varimpnnetDF_impDF)<-c("Importance","Variables")

RF_gg<-ggplot(varImp(object=fs_model_rf))+ggtitle("FS RF - Variable Importance")
GBM_gg<-ggplot(varImp(object=fs_model_gbm))+ggtitle("FS GBM - Variable Importance")
NNET_gg<-ggplot(data=fs_varimpnnetDF_impDF, aes(x=Variables, y=Importance)) + ggtitle("FS NNET - Variable Importance")+
  geom_bar(stat="identity") + coord_flip()
NB_gg<-ggplot(varImp(object=fs_model_nb))+ggtitle("FS NB - Variable Importance")
grid.arrange(RF_gg,NNET_gg,NB_gg,GBM_gg, ncol= 2 )

##Repeating 6 to 8 - end

########## 10. b. Create a dataframe that compares the non-feature selected models ( the same as on 7) and add the best BEST performing models 
#                 of each (randomforest, neural net, naive bayes and gbm) and display the data frame that has the following columns:
#                 ExperimentName, accuracy, kappa. Sort the data frame by accuracy.
########## Answer10-b
comb_summaryDF <- rbind(fs_summarydf, summarydf)
comb_summaryDF=comb_summaryDF[order(-comb_summaryDF$Accuracy),]
rownames(comb_summaryDF) <- NULL
print(comb_summaryDF)

########## 10. c. Which model performs the best? and why do you think this is the case? Can we accurately predict species on this dataset?
########## Answer10-c
# With the below results:
# Model  Accuracy     Kappa
# 1            fs_model_nb 0.7273329 0.5215849
# 2          fs_model_nnet 0.7190828 0.5213849
# 3             model_nnet 0.6904109 0.4932604
# 4            fs_model_rf 0.6825855 0.4629996
# 5               model_nb 0.6545369 0.4188071
# 6               model_rf 0.6489581 0.3753569
# 7 model_gbm_tuneLength20 0.6339706 0.3945501
# 8           fs_model_gbm 0.6157889 0.3624927
# 9              model_gbm 0.6156117 0.3669437
# We can say that best performing model is of Naive Bayes model build with top 3 predictors. As it has the highest accuracy in comparision to other model's accuracy . 
# Hence, we can say that we can predict Species with 75.76% accuracy. 
