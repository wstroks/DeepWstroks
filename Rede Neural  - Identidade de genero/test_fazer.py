# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:45:21 2018

@author: wstro
"""


import pandas as pd

base = pd.read_csv('nomes.csv')
classe= base.iloc[:,1].values
previsores = base.iloc[:,1:3].values


from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(previsores[:,1:4])
previsores[:,1:4]=imputer.transform(previsores[:,1:4])



from sklearn.preprocessing import StandardScaler
escalona= StandardScaler()
previsores=escalona.fit_transform(previsores)

from sklearn.cross_validation import train_test_split

previsores_treinamento , previsores_teste,classe_treinamento, classe_teste=train_test_split(previsores,classe,test_size=0.25, random_state=0)

from sklearn.ensemble import RandomForestClassifier
classificador=RandomForestClassifier(n_estimators=40,criterion='entropy',random_state=0)
classificador.fit(previsores_treinamento,classe_treinamento)

previsoes=classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix,accuracy_score

precisao=accuracy_score(classe_teste,previsoes)

matriz=confusion_matrix(classe_teste,previsoes)
