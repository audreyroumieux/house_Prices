# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:44:41 2018
@author: audrey roumieux
Projet: house Price
Description: Prédiction des prix
"""
#### import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


##### Téléchargement des données
train = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\train.csv')
test = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\test.csv')
sample_submission = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\sample_submission.csv')

#  #%%
##### Visualitation de Données
print(train.describe())
# on observe des données manquente parmis les donné numerique. 
# Des colonnes ont des 0 en median alors qu'elles ont des valeurs moyennes.
 
print(train.info())
# données manquentes: LotFrontage, Alley, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageYrBlt, GarageFinish, GarageQual, GarageCond, PoolQC, Fence, MiscFeature



#  #%%
### Categorisation des features
binaire = ['Street', 'CentralAir']

numerique =['LotFrontage', 'LotArea', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','BedroomAbvGr', 
            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 
            'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
            'PoolArea', 'MiscVal']

categoriel = ['MSSubClass', 'MSZoning','Alley', 'LotShape', 'LandContour', 'Utilities', 
              'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
              'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
              'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 
              'BsmtFinType1', 'BsmtFinType2','Heating', 'Electrical', 'Functional',
              'GarageType', 'GarageFinish', 'PavedDrive', 'Fence', 'MiscFeature', 
              'SaleType', 'SaleCondition']

ordinal = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 
           'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
           'GarageCond','PoolQC']

dates = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']

#lname = list(set(binaire) | set(numerique) |set(categoriel) |set(ordinal) |set(dates))


# #%%

def one_hot_encoding(df):
    for name_column in categoriel:
        # Get one hot encoding on columns
        one_hot = pd.get_dummies(df[name_column])
        # Rename col of dummies
        for name in one_hot:
            one_hot.rename(index=str, columns={name: name_column+"_"+str(name)})
        # Join the encode to df
        pd.concat([df, one_hot], axis=1)
        # Drop columns has it is now encoding
        df = df.drop(name_column, axis=1)
    return df

def na_by_type(df):
    for name_column in categoriel:
        print('nan categoriel: ', df[name_column].isnull().any())
    for name_column in ordinal:
        print('nan ordinal: ', df[name_column].isnull().any())
    for name_column in dates:
        print('nan dates: ', df[name_column].isnull().any())

def numerize_ordinal(df):
    for name in ordinal:
        df[name].replace(['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA'], [0, 1, 2, 3, 4, 5])
    
    
def binaire_in_01(df):
    df['Street'].replace(['Grvl', 'Pave'], [0,1])
    df['CentralAir'].replace(['N','Y'], [0,1])

def nominal_nan(df):
    print("toto")
   
def numerique_nan(df):
    for name in binaire:
        df[name] = np.nan_to_num(df[name])
        
    for name in numerique:
        # remplace les valeurs nan en 0 et infin par val max/min
        df[name] = np.nan_to_num(df[name])
    return df
        
        
def transform_features(df):
    numerize_ordinal(df)
    binaire_in_01(df)
    df = numerique_nan(df)
    na_by_type(df)
    df = one_hot_encoding(df)
    return df


data_train = transform_features(train)
data_test = transform_features(test)

#%%
# On s'assure que l'on a les meme colonne dans le jeux d entrainement et de test
trainColSet = set(data_train.columns.tolist())
testColSet = set(data_test.columns.tolist())
dataCol = list(trainColSet and testColSet)


y = data_train.SalePrice
X = data_train[dataCol]
Xtest = data_test[dataCol]

#%%
##### feature importances
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
forest.fit(X, y)

importances = forest.feature_importances_
std = np.std([arbre.feature_importances_ for arbre in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]


##### Print the feature ranking
#print("Feature ranking:")
X_select_id_col_feature_importance = []
for f in range(X.shape[1]):
    if importances[indices[f]] >= 0.02 :
        print("%d. feature %d = %d (%f)" % (f + 1, indices[f], X.columns[indices[f]], importances[indices[f]]))
        X_select_id_col_feature_importance.append(indices[f])
 
#%%
# Select only  features importance
X_col_feature_importance = []
for id in X_select_id_col_feature_importance:
    X_col_feature_importance.append(X.columns[id])
    
X_train_less_col = X[X_col_feature_importance]
X_test_less_col = Xtest[X_col_feature_importance]

#%%
##### Split des données Train/Test
x_train, x_test, y_train, y_test = train_test_split(X_train_less_col, y, test_size=.3, random_state=42)

#%%
"""
#chose the Best paramaters
from sklearn.model_selection import GridSearchCV

model = KNeighborsRegressor()
parameters = {'n_neighbors' : [value for value in range(2, 10)]}
grid = GridSearchCV(model, param_grid = parameters, cv = 3)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)


model = tree.DecisionTreeRegressor()
parameters = {'max_depth': [value for value in range(2, 10)] ,
            'max_features' : [value for value in range(2, 10)]}
grid = GridSearchCV(model, param_grid = parameters, cv = 3)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)

model = RandomForestRegressor()
parameters = {'n_estimators': [value for value in range(2, 10)],
              'max_depth': [value for value in range(2, 10)], 
              'max_features' : [value for value in range(2, 10)]}
grid = GridSearchCV(model, param_grid = parameters, cv = 3)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
"""

#%%
##### Entrainement (descente de gradien, affichage d'erreur)
def entrainement():
    listEnt = []
    listEnt.append(linear_model.LinearRegression().fit(x_train, y_train))
    listEnt.append(tree.DecisionTreeRegressor(max_depth=5, max_features=9).fit(x_train, y_train))
    listEnt.append(KNeighborsRegressor(n_neighbors=6).fit(x_train, y_train))
    listEnt.append(RandomForestRegressor(n_estimators=8 , max_features=6 , max_depth=8).fit(x_train, y_train))
    listEnt.append(linear_model.Lasso(alpha=0.1).fit(x_train, y_train))
    listEnt.append(linear_model.Ridge(alpha=1.0).fit(x_train, y_train))
    #listEnt.append(PolynomialFeatures().fit(x_train, y_train))
    return listEnt

entr = entrainement()

#%%
##### Verification de la performance sur le jeu de donnée test
print("\n Linear Regression : ")
print('Train score: ', entr[0].score(x_train, y_train))
print('Test score: ', entr[0].score(x_test, y_test))

print("\n Decision Tree : ")
print('Train score: ', entr[1].score(x_train, y_train))
print('Test score: ', entr[1].score(x_test, y_test))

print("\n K Neighbors : ")
print('Train score: ', entr[2].score(x_train, y_train))
print('Test score: ', entr[2].score(x_test, y_test))

print("\n Random Forest : ")
print('Train score: ', entr[3].score(x_train, y_train))
print('Test score: ', entr[3].score(x_test, y_test))

print("\n regression lasso : ")
print('Train score: ', entr[4].score(x_train, y_train))
print('Test score: ', entr[4].score(x_test, y_test))

print("\n regression ridge : ")
print('Train score: ', entr[5].score(x_train, y_train))
print('Test score: ', entr[5].score(x_test, y_test))

#%%  
#### Récuperation des données a mettre sur kaagle
predictTest = entr[3].predict(X_test_less_col)

my_submission = pd.DataFrame({'Id': Xtest.Id, 'SalePrice': predictTest}) 
#my_submission.to_csv('submissionT.csv', index = False)
