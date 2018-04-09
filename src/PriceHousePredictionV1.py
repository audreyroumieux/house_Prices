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

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

#%%
##### Téléchargement des données
train = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\train.csv')
test = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\test.csv')
sample_submission = pd.read_csv('C:\\Users\\b-auroum\\Desktop\\Work\\PROJET\\house_Prices\\data\\sample_submission.csv')

#%%
##### Affichage des données
"""
print(train.describe())
# on observe des données manquente parmis les donné numerique. 
# Des colonnes ont des 0 en median alors qu'elles ont des valeurs moyennes.
 
print(train.info())
# données manquentes: LotFrontage, Alley, BsmtExposure, BsmtFinType1, BsmtFinType2, FireplaceQu, GarageYrBlt, GarageFinish, GarageQual, GarageCond, PoolQC, Fence, MiscFeature

print(train.head())
print(type(train))

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
"""
#%%
##### Prétraitement des données et Création de nouvelle variable
def type_category(df):
    binaire = pd.DataFrame()
    numerique = pd.DataFrame()
    nominal = pd.DataFrame()
    
    for column in df :
        if len(df[column].value_counts()) == 2 :
            print("var binnaire")
            encoder = LabelEncoder()
            try :
                df[column] = encoder.fit_transform(df[column].astype(str))
                binaire.assign(column = df[column])
                print()
            except :
                print('Error binnary encoding '+ encoder)
               
        elif (df[column].dtype == "float") or (df[column].dtype == "int64"):
            print("var numerique")
            numerique.assign(column = df[column])
           
        else : #dtype == 'category','object'
            print("var ordinal/categorielle")
            nominal.assign(column = df[column])
            
    typeCategory = [ binaire, numerique, nominal]           
    return typeCategory


def replace_one_hot_encoding(df):
    columnsToEncode = df.select_dtypes(include=['category','object'])
    # df.dtypes[train.dtypes=="object"].index.values
    
    for column in columnsToEncode: 
        #s'il y a trops de category differente dans column, alors on ne traitera pas ces données exp: name_colonne
        if len(df[column].value_counts()) > 10 : 
            df = df.drop([column], axis=1)
        # sinon on fait du one hot encoding dessus
        else :
            # Get one hot encoding on columns
            one_hot = pd.get_dummies(df[column])
            # Drop columns has it is now encoding
            df = df.drop([column], axis=1)
            # Join the encode to df
            df = df.join(one_hot)
            
    return df

def haveGarage(df):
    df['haveGarage'] = df['GarageType'].isna()
    return df


def havePool(df):
    df['havePool'] = df['PoolArea'].isna()
    return df

def drop_features(df):
    na_columns=train.columns[data_train.isna().any()]
    print(data_train[na_columns].isna().sum())
    return df.drop(['Id'], axis=1)


def transform_features(df):
    df = haveGarage(df)
    df = havePool(df)
    #df = replace_one_hot_encoding(df)
    #df = drop_features(df)
    return df


data_train = transform_features(train)
data_test = transform_features(test)
#data_train.head()

typeCategory = type_category(train)

#%%
#### Renomage des colonnes
'''
data_train.rename(index=str, 
             columns={'MSSubClass': 'BuildingClass', 'MSZoning': 'ZoningClass',
                      'LotArea': 'Aire_m²', 'RoofMatl': 'RoofMaterial', 
                      'ScreenPorch' : 'verenda_m²', 'PoolArea':'Piscine_m²', 
                      'PoolQC':'PiscineQualite', 'Fence':'BarriereQualite',
                      'MiscFeature': 'Divers', 'MiscVal':'ValueDivers', 
                      'MoSold': 'MoisVendu', 'YrSold':'AnneeVendu'})
#print(train.columns)
'''
#%%
##### feature importances
y = data_train.SalePrice
X = data_train.drop(['Id', 'SalePrice', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)
Xtest = data_test.drop(['Id', 'LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)

#%%
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators = 250, random_state = 0)
forest.fit(X, y)

importances = forest.feature_importances_
std = np.std([arbre.feature_importances_ for arbre in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

X_select_id_col_feature_importance = []
for f in range(X.shape[1]):
    if importances[indices[f]] >= 0.02 :
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        X_select_id_col_feature_importance.append(indices[f])
        
print() 

#%%
X_col_feature_importance = []
Xtest_col_feature_importance = []
for id in X_select_id_col_feature_importance:
    X_col_feature_importance.append(X.columns[id])
    Xtest_col_feature_importance.append(X.columns[id])
    
X_test_less_col = Xtest[Xtest_col_feature_importance]


##### Split des données Train/Test
x_train, x_test, y_train, y_test = train_test_split(X[X_col_feature_importance], y, test_size=.3, random_state=42)
# print(x_train.head())
# print(x_train.describe())

#%%
##### Entrainement (descente de gradien, affichage d'erreur)
def entrainement():
    listEnt=[]
    listEnt.append(linear_model.LinearRegression().fit(x_train, y_train))
    listEnt.append(tree.DecisionTreeRegressor().fit(x_train, y_train))
    listEnt.append(KNeighborsRegressor(n_neighbors=5).fit(x_train, y_train))
    listEnt.append(RandomForestRegressor(max_depth=2, random_state=0).fit(x_train, y_train))
    return listEnt

entr = entrainement()
print(entr)

#%%
##### Verification de la performance sur le jeu de donnée test
print("Train : ")
for i in entr:
    print(i.score(x_train, y_train))
    
print("Test : ")
for i in entr:
    print(i.score(x_test, y_test))

#%%  
#### Récuperation des données a mettre sur kaagle
na_columns_test = X_test_less_col.columns[X_test_less_col.isna().any()]
print(X_test_less_col[na_columns_test].isna().sum())

regKNN = KNeighborsRegressor(n_neighbors=5)
regKNN.fit(x_train, y_train)
print(x_train.columns)
print(Xtest.columns)
predictTest = regKNN.predict(Xtest)

#%%
my_submission = pd.DataFrame({'Id': Xtest.Id, 'SalePrice': predictTest}) 
#my_submission.to_csv('submissionT.csv', index = False)
my_submission