
import os

os.getcwd()
os.chdir("C:/Users/gopin/Documents/R/BikeRental-Project")

import pandas as pd
import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("day.csv")

savedData = data

dataCasual = savedData[["season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual"]]
dataRegsitered = savedData[["season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","registered"]]

dCasual = dataCasual.copy()
dRegistered = dataRegsitered.copy()

# Store continuous variable names
cnames_C = ["season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual"]
cnames_R = ["season","yr","mnth","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","registered"]

# Detect outliers & delete(Casual)
for i in cnames_C:
    q75, q25 = np.percentile(dCasual.loc[:,i],[75,25])
    iqr = q75 - q25
    innerfence = q25 - (iqr * 1.5)
    outerfence = q75 + (iqr * 1.5)
    dCasual = dCasual.drop(dCasual[dCasual.loc[:,i] < innerfence].index)
    dCasual = dCasual.drop(dCasual[dCasual.loc[:,i] > outerfence].index)
# Replace with NA
dCasual = dataCasual.copy()
for i in cnames_C:
    q75, q25 = np.percentile(dCasual.loc[:,i],[75,25])
    iqr = q75 - q25
    innerfence = q25 - (iqr * 1.5)
    outerfence = q75 + (iqr * 1.5)
    dCasual.loc[dCasual[i] < innerfence,:i] = np.nan
    dCasual.loc[dCasual[i] > outerfence,:i] = np.nan
# Calculate Missing values
missing_C = pd.DataFrame(dCasual.isnull().sum())
# Impute using Mode method
for i in cnames_C:
    dCasual[i] = dCasual[i].fillna(dCasual[i].mode()[0])
#missing_C = pd.DataFrame(dCasual.isnull().sum())
savedDataCasual = dCasual

# Detect outliers & delete (Registered)
for i in cnames_R:
    q75, q25 = np.percentile(dRegistered.loc[:,i],[75,25])
    iqr = q75 - q25
    innerfence = q25 - (iqr * 1.5)
    outerfence = q75 + (iqr * 1.5)
    dRegistered = dRegistered.drop(dRegistered[dRegistered.loc[:,i] < innerfence].index)
    dRegistered = dRegistered.drop(dRegistered[dRegistered.loc[:,i] > outerfence].index)
# Replace with NA
dRegistered = dRegistered.copy()
for i in cnames_R:
    q75, q25 = np.percentile(dRegistered.loc[:,i],[75,25])
    iqr = q75 - q25
    innerfence = q25 - (iqr * 1.5)
    outerfence = q75 + (iqr * 1.5)
    dRegistered.loc[dRegistered[i] < innerfence,:i] = np.nan
    dRegistered.loc[dRegistered[i] > outerfence,:i] = np.nan
# Calculate Missing values
missing_R = pd.DataFrame(dRegistered.isnull().sum())
# Impute using Mode method
for i in cnames_R:
    dRegistered[i] = dRegistered[i].fillna(dRegistered[i].mode()[0])

savedDataRegistered = dRegistered

# Feature selection
'''import seaborn as sns
from scipy.stats import chi2_contingency
from random import randrange,uniform
for i in dCasual.columns:
    print(i)
    p = chi2_contingency(pd.crosstab(dCasual['casual'],dCasual[i]))

'''
# Gives Barplot
# %matplotlib inline
# plt.hist(dCasual['weekday'], bins='auto')

# Sampling using Systematic sampling
simpleRandomSampling_C = dCasual.sample(100)
simpleRandomSampling_R = dRegistered.sample(100)

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Train and Test data
dCasual = savedDataCasual.copy()
xc = dCasual.values[:, 0:11]
yc = dCasual.values[:, 11]
xc_train, xc_test, yc_train, yc_test = train_test_split(xc,yc,test_size=0.2)

import statsmodels.api as sm
train_c, test_c = train_test_split(dCasual,test_size=0.2)
model_C = sm.OLS(train_c.iloc[:,11], train_c.iloc[:,0:11]).fit()
model_C.summary()

dRegistered = savedDataRegistered.copy()
xr = dRegistered.values[:, 0:11]
yr = dRegistered.values[:, 11]
xr_train, xr_test, yr_train, yr_test = train_test_split(xr,yr,test_size=0.2)

import statsmodels.api as sm
train_c, test_c = train_test_split(dRegistered,test_size=0.2)
model_R = sm.OLS(train_c.iloc[:,11], train_c.iloc[:,0:11]).fit()
model_R.summary()

import ggplot
from ggplot import *
dC = savedDataCasual.copy()

ggplot(dC, aes(x='season', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Season") + ylab("Casual") + ggtitle("Barplot_seasonVScasual") + theme.bw()

ggplot(dC, aes(x='holiday', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Holiday") + ylab("Casual") + ggtitle("Barplot_holidayVScasual") + theme.bw()
    
ggplot(dC, aes(x='mnth', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Month") + ylab("Casual") + ggtitle("Barplot_monthVScasual") + theme.bw()

ggplot(dC, aes(x='weather', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Weather") + ylab("Casual") + ggtitle("Barplot_weatherVScasual") + theme.bw()
    
ggplot(dC, aes(x='weekday', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Weekday") + ylab("Casual") + ggtitle("Barplot_weekDayVScasual") + theme.bw()
    
ggplot(dC, aes(x='workkingday', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Working Day") + ylab("Casual") + ggtitle("Barplot_workingDayVScasual") + theme.bw()
    
ggplot(dC, aes(x='season', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Season") + ylab("Casual") + ggtitle("Barplot_seasonVScasual") + theme.bw()


dR = savedDataRegistered.copy()
ggplot(dR, aes(x='season', y='registered')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Season") + ylab("Registered") + ggtitle("Barplot_seasonVSregistered") + theme.bw()

ggplot(dR, aes(x='holiday', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Holiday") + ylab("Registered") + ggtitle("Barplot_holidayVSregistered") + theme.bw()
    
ggplot(dR, aes(x='mnth', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Month") + ylab("Registered") + ggtitle("Barplot_monthVSregistered") + theme.bw()

ggplot(dR, aes(x='weather', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Weather") + ylab("Registered") + ggtitle("Barplot_weatherVSregistered") + theme.bw()
    
ggplot(dR, aes(x='weekday', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Weekday") + ylab("Registered") + ggtitle("Barplot_weekDayVSregistered") + theme.bw()
    
ggplot(dR, aes(x='workkingday', y='casual')) +\
    geom_bar(fill="blue") +\
    scale_color_brewer(type="diverging", palette=4) +\
    xlab("Working Day") + ylab("Registered") + ggtitle("Barplot_workingDayVSregistered") + theme.bw()
    





