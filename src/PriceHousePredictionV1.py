# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:04:10 2018
@author: audrey roumieux
Projet: house Price
Description: Prédiction des prix
"""
#### import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

#%%
##### Téléchargement des données
train = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\train.csv')
test = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\test.csv')
sample_submission = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\sample_submission.csv')

#%%
##### Affichage des données

print(train.describe())
# on observe des données manquente parmis les donné numerique. les colonnes sont LotFrontage, MasVnrArea 
# MasVnrArea, WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch, PoolArea, MiscVal, MoSold :ont des 0 en median
 
print(train.info())
# données manquentes: LotFrontage, Alley, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageYrBlt, GarageFinish, GarageQual, GarageCond, PoolQC, Fence, MiscFeature

print(train.head())
print(type(train))

print(train.isna().sum())


#%%
plt.figure()
ax = sns.barplot('MSSubClass', 'SalePrice', data = train) 
ax.set(xlabel = 'building Class', ylabel = 'Sale Price')
plt.show()

#il manque l'element Po =poor
plt.figure()
ax = sns.barplot('KitchenQual', 'SalePrice', data = train) 
ax.set(xlabel = 'Kitchen Quality', ylabel = 'Sale Price')
plt.show()

print("number of Kitchen: ")
print(train['KitchenAbvGr'].value_counts())


plt.figure()
plt.scatter(train["PoolArea"], train['SalePrice'])
plt.show()

# histograme
plt.figure()
ax = train.BedroomAbvGr.hist(bins=15, color='teal', alpha=0.8)
ax.set(xlabel='number of Bedroom', ylabel='Count')
plt.show()



plt.figure()
ax = train.GarageType.hist(bins=15, color='teal', alpha=0.8)
ax.set(xlabel='Type de Garage', ylabel='Count')
plt.show()

#%%
##### Prétraitement des données et Création de nouvelle variable
    
def haveGarage(df):
    df['haveGarage'] = df['GarageType'].isna()
    return df

def havePool(df):
    df['havePool'] = df['PoolArea'].isna()
    return df

def replace(df):
    
    for col in df.dtypes[df.dtypes=="object"].index.values:
        df[col]
    
    #df['GarageType'].replace(to_replace=dict(2Types=6, Attchd=5, Basment=4, BuiltIn=3, CarPort=2, Detchd=1), inplace=True)
    df['MSZoning'].replace(to_replace = dict(A=7, C=6, FV=5, I=4, RH=3, RL=2, RP=1, RM=0), inplace=True)
    df['Street'].replace(to_replace = dict(Pave=0, Grvl=1), inplace=True)
    df['LotShape'].replace(to_replace = dict(Reg=3, IR1=2, IR2=1, IR3=0), inplace=True)
    df['LandContour'].replace(to_replace = dict(Lvl=3, Bnk=2, HLS=1, Low=0), inplace=True)
    #df['BldgType'].replace(to_replace = dict(1Fam= 1, 2FmCon = 2, Duplx = 3, TwnhsE = 4, TwnhsI = 5), inplace = True)
    df.replace(to_replace = dict(Ex=5, Gd=4, TA=3, Fa=2, Po=1), inplace=True)
    return df

def drop_features(df):
    return df.drop(['Id'], axis=1)


def transform_features(df):
    df = haveGarage(df)
    df = havePool(df)
    df = replace(df)
    df = drop_features(df)
    return df


data_train = transform_features(train)
data_test = transform_features(test)
data_train.head()

for col in ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']:
    train[col] = data_train[col].astype('category')
    train[col].cat.codes
    
#%%
#### Renomage des colonnes
data_train.rename(index=str, 
             columns={'MSSubClass': 'BuildingClass', 'MSZoning': 'ZoningClass',
                      'LotArea': 'Aire_m²', 'RoofMatl': 'RoofMaterial', 
                      'ScreenPorch' : 'verenda_m²', 'PoolArea':'Piscine_m²', 
                      'PoolQC':'PiscineQualite', 'Fence':'BarriereQualite',
                      'MiscFeature': 'Divers', 'MiscVal':'ValueDivers', 
                      'MoSold': 'MoisVendu', 'YrSold':'AnneeVendu'})
print(train.columns)

y = data_train.SalePrice
#%%
##### feature importances
y = data_train.SalePrice
X = data_train[['MSSubClass', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'OverallQual', 'OverallCond', 'YearBuilt', 'PoolArea', 'havePool', 'haveGarage']]


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
#%%
forest.fit(X, y)

#%%
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
#%%
##### Split des données Train/Test


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
print(x_train.head())
print(x_train.describe())

#%%
##### Entrainement (descente de gradien, affichage d'erreur)
def entrainement():
    listEnt=[]
    listEnt.append(linear_model.LinearRegression().fit(x_train, y_train))
    listEnt.append(tree.DecisionTree...().fit(x_train, y_train))
    listEnt.append(KNeighbors...().fit(x_train, y_train))
    
    return listEnt

#entr = entrainement()

#%%
##### Verification de la performance sur le jeu de donnée test