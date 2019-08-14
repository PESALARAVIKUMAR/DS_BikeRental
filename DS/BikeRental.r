

rm(list=ls())
install.packages(c("dmm","dplyr","plyr","reshape","ggplot2","data.table","psych","usdm","caret","DMwR"))
data=read.csv("day.csv",header=T)
newData=subset(data, select = c("season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual","registered","cnt"))
savedData=newData
dataCasual = subset(newData, select = c("season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual"))
saved_dataCasual = dataCasual;
dataRegistered = subset(newData, select = c("season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","registered"))
saved_dataRegistered = dataRegistered
write.csv(dataCasual, "dataCasual.csv",row.names = T)
write.csv(dataRegistered, "dataRegistered.csv",row.names = T)


# Retrive numeric data
numeric_index_casual = sapply(dataCasual,is.numeric)
numeric_data_casual = dataCasual[,numeric_index_casual]
numeric_data_cols_casual = colnames(numeric_data_casual)

numeric_index_registered = sapply(dataRegistered,is.numeric)
numeric_data_registered = dataRegistered[,numeric_index_registered]
numeric_data_cols_registered = colnames(numeric_data_registered)

#Calculate outliers in dataCasual
dataCasual1 = dataCasual
for(i in numeric_data_cols_casual)
{
  print(i)
  val_casual = dataCasual1[,i][dataCasual1[,i]%in%boxplot.stats(dataCasual1[,i])$out]
  print(length(val_casual))
  dataCasual1 = dataCasual1[which(!dataCasual1[,i]%in%val_casual),]
}
# Replace all outliers in dataCasual with NA and impute missing using missing value analysis
dataCasual1 = dataCasual
for(i in numeric_data_cols_casual)
{
  val_casual = dataCasual1[,i][dataCasual1[,i]%in%boxplot.stats(dataCasual1[,i])$out]
  dataCasual1[,i][dataCasual1[,i]%in%val_casual] = NA
}

#Calculate outliers in dataRegistered
dataRegistered1 = dataRegistered
for(i in numeric_data_cols_registered)
{
  print(i)
  val_registered = dataRegistered1[,i][dataRegistered1[,i]%in%boxplot.stats(dataRegistered1[,i])$out]
  print(length(val_registered))
  dataRegistered1 = dataRegistered1[which(!dataRegistered1[,i]%in%val_registered),]
}
# Replace all outliers in dataRegistered with NA and impute missing using missing value analysis
dataRegistered1 = dataRegistered
for(i in numeric_data_cols_registered)
{
  val_registered=dataRegistered1[,i][dataRegistered1[,i]%in%boxplot.stats(dataRegistered1[,i])$out]
  dataRegistered1[,i][dataRegistered1[,i]%in%val_registered] = NA
}

# Apply KNN imputation for dataCasual
require(DMwR)
dataCasual1 = knnImputation(dataCasual1,k=5)

# Apply Mean imputation for columns with NA (because of error: there are not sufficient cases)
dataCasual1$holiday[is.na(dataCasual1$holiday)] = mean(dataCasual1$holiday,na.rm = T)
dataCasual1$hum[is.na(dataCasual1$hum)] = mean(dataCasual1$hum,na.rm = T)
dataCasual1$windspeed[is.na(dataCasual1$windspeed)] = mean(dataCasual1$windspeed,na.rm = T)
dataCasual1$casual[is.na(dataCasual1$casual)] = mean(dataCasual1$casual,na.rm = T)

# Apply KNN imputation for dataRegistered
dataRegistered1 = knnImputation(dataRegistered1,k=5)

# Apply Mean imputation for columns with NA (because of error: there are not sufficient cases)
dataRegistered1$holiday[is.na(dataRegistered1$holiday)] = mean(dataRegistered1$holiday,na.rm = T)
dataRegistered1$hum[is.na(dataRegistered1$hum)] = mean(dataRegistered1$hum,na.rm = T)
dataRegistered1$windspeed[is.na(dataRegistered1$windspeed)] = mean(dataRegistered1$windspeed,na.rm = T)
dataRegistered1$registered[is.na(dataRegistered1$registered)] = mean(dataRegistered1$registered,na.rm = T)


# chi-square test of Independency
# factor_index=sapply(dataCasual1, is.factor)
# factor_data=dataCasual1[,factor_index]
# 
# for (i in 1:ncol(dataCasual1)-1)
# {
#   print(chisq.test(table(dataCasual1$dataCasual1[,length(dataCasual1)],dataCasual1[,i]))) 
# }

# select Dependent columns from dataCasual
nonDependentCols_casual = names(dataCasual1)%in% c("casual")
dependentData_casual = dataCasual1[!nonDependentCols_casual]
dependentCols_casual = colnames(dependentData_casual)
# select Dependent columns from dataRegistered
nonDependentCols_registered = names(dataRegistered1)%in% c("registered")
dependentData_registered = dataRegistered1[!nonDependentCols_registered]
depedentCols_registered = colnames(dependentData_registered)

# Feature Scaling
# require(graphics)
# for(i in dependentCols_casual)
# {
#   print(i)
#   qqnorm(dependentData_casual$i)
#   hist(dependentData_casual$i)
# }

# Sampling Techniques
d_dataCasual = dependentData_casual
simpleRandomSampling_dataCasual = d_dataCasual[sample(nrow(d_dataCasual),100,replace = F),]
d_dataRegistered = dependentData_registered
simpleRandomSampling_dataRegistered = d_dataRegistered[sample(nrow(d_dataRegistered),100,replace = F),]

# # Decision tree classifier for dataRegistered
# # divide train & test data
train_index_casual = sample(1:nrow(saved_dataCasual), 0.8 * nrow(saved_dataCasual), prob = NULL)
train_data_casual = saved_dataCasual[train_index_casual,]
test_data_casual = saved_dataCasual[-train_index_casual,]
train_index_registered = sample(1:nrow(saved_dataRegistered), 0.8 * nrow(saved_dataRegistered), prob = NULL)
train_data_registered = saved_dataRegistered[train_index_registered,]
test_data_registered = saved_dataRegistered[-train_index_registered,]
# # builds decision tree
# library(rpart)
# fit = rpart(registered~., data=train_data_registered, method="anova")
# library(MASS)
# predictions = predict(fit, test_data_registered[,-12])
# # Calculate MAPE, MSE, RMSE, MAE
# library(DMwR)
# regr.eval(test_data_registered[,12], predictions, stats = c('mae','mape','mse','rmse'))
# mae         mape          mse         rmse 
# 5.827114e+02 2.279066e+00 7.503266e+05 8.662140e+02 


# KNN 
# install.packages("caret")
# train_index_casual = sample(1:nrow(saved_dataCasual), 0.8 * nrow(saved_dataCasual), prob = NULL)
# train_data_casual = saved_dataCasual[train_index_casual,]
# test_data_casual = saved_dataCasual[-train_index_casual,]
# train_index_registered = sample(1:nrow(saved_dataRegistered), 0.8 * nrow(saved_dataRegistered), prob = NULL)
# train_data_registered = saved_dataRegistered[train_index_registered,]
# test_data_registered = saved_dataRegistered[-train_index_registered,]
# library(class)
# knn_predictions_casual = knn(train_data_casual[,1:11], test_data_casual[,1:11], train_data_casual$casual, k=5)
# knn_predictions_registered = knn(train_data_registered[,1:11], test_data_registered[,1:11], train_data_registered$registered, k=5)
# #Accuracy
# knn_CM_casual = table(knn_predictions_casual, test_data_casual$casual)
# sum(diag(knn_CM_casual))/nrow(test_data_casual)
# TN_casual = knn_CM_casual[0,0]
# FN_casual = knn_CM_casual[1,0]
# TP_casual = knn_CM_casual[1,1]
# FP_casual = knn_CM_casual[0,1]
# 
# knn_CM_registered = table(knn_predictions_registered, test_data_registered$registered)
# sum(diag(knn_CM_registered))/nrow(test_data_registered)
# TN_registered = knn_CM_registered[0,0]
# FN_registered = knn_CM_registered[1,0]
# TP_registered = knn_CM_registered[1,1]
# FP_registered = knn_CM_registered[0,1]
# 
# library(DMwR)
# regr.eval(test_data_casual[,12], knn_predictions_casual, stats = c('mae','mape','mse','rmse'))
# regr.eval(test_data_registered[,12], knn_predictions_registered, stats = c('mae','mape','mse','rmse'))

# Linear Regression (Registered)
install.packages("usdm")
library(usdm)

vif(saved_dataCasual[,-12])
vifcor(saved_dataCasual[,-12], th = 1.0)
lr_model_casual = lm(casual~., data = saved_dataCasual)
summary(lr_model_casual)
lr_prediction_casual = predict(lr_model_casual, test_data_casual[,1:11])
library(DMwR)
library(MASS)
regr.eval(test_data_registered[,12], lr_prediction_casual, stats = c('mae','mape','mse','rmse'))


vif(saved_dataRegistered[,-12])
vifcor(saved_dataRegistered[,-12], th = 1.0)
lr_model_registered = lm(registered~., data = saved_dataRegistered)
summary(lr_model_registered)
lr_prediction_registered = predict(lr_model_registered, test_data_casual[,1:11])
library(DMwR)
library(MASS)
regr.eval(test_data_registered[,12], lr_prediction_registered, stats = c('mae','mape','mse','rmse'))


# KMeans Clustering
install.packages("NbClust")
library(NbClust)
d_casual = saved_dataCasual
clusters_casual = NbClust(d_casual, min.nc = 2, max.nc = 10, method = "kmeans")
barplot(table(clusters_casual$Best.nc[1,]), xlab="X", ylab="Y", main="")
kmeans_model_casual = kmeans(d_casual,4,nstart = 25)
cluster_accuracy_casual = table(d_casual$casual,kmeans_model_casual$cluster)

d_registered = saved_dataRegistered
clusters_registered = NbClust(d_registered, min.nc = 2, max.nc = 10, method = "kmeans")
barplot(table(clusters_registered$Best.nc[1,]), xlab="X", ylab="Y", main="")
kmeans_model_registered = kmeans(d_registered,4,nstart = 25)
cluster_accuracy_registered = table(d_registered$registered,kmeans_model_registered$cluster)

library(ggplot2)
library(scales)
library(psych)
library(gplots)

newData_casual = dataCasual
newData_registered = dataRegistered

# Bar plot ( Categorical variables VS Target variable)
# Casual Data
ggplot(newData_casual, aes_string(x=newData_casual$season, y=newData_casual$casual)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("season") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Season vs Casual ") + theme(text = element_text(size=10))

ggplot(newData_casual, aes_string(x=newData_casual$mnth, y=newData_casual$casual)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("month") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Month vs Casual ") + theme(text = element_text(size=10))

ggplot(newData_casual, aes_string(x=newData_casual$holiday, y=newData_casual$casual)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("holiday") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Holiday vs Casual ") + theme(text = element_text(size=10))

ggplot(newData_casual, aes_string(x=newData_casual$weekday, y=newData_casual$casual)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("weekday") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Weekday vs Casul ") + theme(text = element_text(size=10))

ggplot(newData_casual, aes_string(x=newData_casual$workingday, y=newData_casual$casual)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("workingday") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Working Day vs Casual") + theme(text = element_text(size=10))

ggplot(newData_casual, aes_string(x=newData_casual$weathersit, y=newData_casual$casual)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("Weather") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot weather vs casual ") + theme(text = element_text(size=10))

# Registered Data
ggplot(newData_registered, aes_string(x=newData_registered$season, y=newData_registered$registered)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("season") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Season vs Registered ") + theme(text = element_text(size=10))

ggplot(newData_registered, aes_string(x=newData_registered$mnth, y=newData_registered$registered)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("month") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Month vs Registered ") + theme(text = element_text(size=10))

ggplot(newData_registered, aes_string(x=newData_registered$holiday, y=newData_registered$registered)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("holiday") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Holiday vs Registered ") + theme(text = element_text(size=10))

ggplot(newData_casual, aes_string(x=newData_casual$weekday, y=newData_casual$casual)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("weekday") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Weekday vs Registered ") + theme(text = element_text(size=10))

ggplot(newData_registered, aes_string(x=newData_registered$workingday, y=newData_registered$registered)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("workingday") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Working Day vs Registered") + theme(text = element_text(size=10))

ggplot(newData_registered, aes_string(x=newData_registered$weathersit, y=newData_registered$registered)) +
  geom_bar(stat = "identity",fill="Blue") + theme_bw() +
  xlab("Weather") + ylab("casual") + 
  scale_y_continuous(breaks = pretty_breaks(n=10)) + 
  ggtitle("Bar plot Weather vs Registered ") + theme(text = element_text(size=10))

















