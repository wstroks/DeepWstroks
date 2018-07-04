# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:45:22 2018

@author: wstro
"""


import pandas as ps

base=ps.read_csv('credit-data.csv')
base.loc[base.age <0,'age']=40.92

previsores=base.iloc[:,1:4].values
classe=base.iloc[:,4].values

from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(previsores[:,1:4])
previsores[:,1:4]=imputer.transform(previsores[:,1:4])


from sklearn.preprocessing import StandardScaler
escalona= StandardScaler()
previsores=escalona.fit_transform(previsores)

from sklearn.cross_validation import train_test_split

previsores_treinamento , previsores_teste,classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier
classificador=DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento,classe_treinamento)

previsoes=classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix,accuracy_score

precisao=accuracy_score(classe_teste,previsoes)

matriz=confusion_matrix(classe_teste,previsoes)
