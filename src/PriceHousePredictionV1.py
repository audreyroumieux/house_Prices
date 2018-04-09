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
##### Prétraitement des données et Création de nouvelle variable
def rename_na_by_no (df):
    
    #df.BsmtHalfBath = df.BsmtHalfBath.fillna(df.BsmtHalfBath.median())
    numerique = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']
    for name in numerique:
        df[name] = np.nan_to_num(df[name])

    for col in df.columns[df.isnull().any()].tolist():
        df[col] = df[col].fillna("no"+col)
        
    """
    df['Alley'] = df['Alley'].fillna('noAlleyAccess')
    df['BsmtQual'] = df['BsmtQual'].fillna('noBasement')
    df['BsmtCond'] = df['BsmtCond'].fillna('noBasement')
    df['BsmtExposure'] = df['BsmtExposure'].fillna('noBasement')
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('noBasement')
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('noBasement')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('noFireplace')
    #df['haveGarage'] = df['GarageType'].isna()
    df['GarageType'] = df['GarageType'].fillna('noGarage')
    df['GarageFinish'] = df['GarageFinish'].fillna('noGarage')
    df['GarageQual'] = df['GarageQual'].fillna('noGarage')
    df['GarageCond'] = df['GarageCond'].fillna('noGarage')
    #df['havePool'] = df['PoolArea'].isna()
    df['PoolArea'] = df['PoolArea'].fillna('noPool')
    df['PoolQC'] = df['PoolQC'].fillna('noPool')
    df['Fence'] = df['Fence'].fillna('noFence')
    """
    return df


def colNa(df):
    choices = ['yes', 'no', 'y', 'n', 'oui', 'non']
    valid = False
    while valid == False:
        answer = input("Es tu sure d avoir traiter les colonnes avec des Na? (yes or no)" ).strip()
        valid =  answer in choices 
    print("Car il reste : ", df.columns[df.isnull().any()].tolist())


#
def type_category(df):
    binaire = pd.DataFrame()
    numerique = pd.DataFrame()
    nominal = pd.DataFrame()
    
    for column in df :
        if len(df[column].value_counts()) == 2 :
            encoder = LabelEncoder()
            try :
                df[column] = encoder.fit_transform(df[column].astype(str))
                binaire = pd.concat([binaire , df[column]], axis=1)
                #print("var % binaire", column)
                
            except :
                print('Error binnary encoding '+ encoder)
                
        elif (df[column].dtype == "float"): # or (df[column].dtype == "int"):
            #print("var % numerique", column)
            numerique = pd.concat([numerique , df[column]], axis=1)
            
        elif len(df[column].value_counts()) <= len(df[column]):
            #print("var % nominal", column)
            nominal = pd.concat([nominal , df[column]], axis=1)
           
        else : #dtype == 'category','object'
            print("VAR NON TRAITEE !!!!!!!", column, "type ", df[column].dtype)
            #nominal = pd.concat([nominal , df[column]], axis=1)
            
    typeCategory = {'Binaire': binaire, 'Numerique': numerique, 'Nominal': nominal}           
    return typeCategory


def one_hot_encoding(df):
    for column in df:  
        if df[column].dtype == 'object':
            # Get one hot encoding on columns
            one_hot = pd.get_dummies(df[column])
            
            #rename
            for name in one_hot:
                #print(column+"_"+name)
                one_hot.rename(index=str, columns={name: column+"_"+name})
            # Join the encode to df
            pd.concat([df, one_hot], axis=1)
            #df = df.join(one_hot)
            # Drop columns has it is now encoding
            df = df.drop(column, axis=1)
            
    return df


def drop_features(df):
    if 'Id' in df.columns:
        df = df.drop(['Id'], axis=1)
    return df


def transform_features(df):
    rename_na_by_no(df)
    colNa(df)
    #df = drop_features(df)
    typeCategory = type_category(df)
    df2 = one_hot_encoding(typeCategory['Nominal'])
    return df2


data_train = transform_features(train)
data_test = transform_features(test)
#data_train.head()

#%%
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

#%%
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

print()
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

