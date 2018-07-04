# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:51:56 2018

@author: wstroks
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')

base.describe()#ele retorna algumas estatisticas de cada atributo

base.loc[base['age']< 0]
#apagar a coluna age
#base.drop('age',1,inplace=True)

#apagar somente os registros com problemas
base.drop(base[base.age < 0].index,inplace=True)

#preencher os valores manualmente

#preencher os valores com a média
base.mean()#tras a media
base.age.mean()#media de idade leva em consideração os erros -28 de idade ex
base['age'][base.age > 0].mean()#tirando os numeros negativos da media
base.loc[base.age < 0, 'age']=40.92



pd.isnull(base['age'])#retorna e existe algum valor nulo ... ou seja true
base.loc[pd.isnull(base['age'])]#retorna só os valores nulos

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values
#tirar nan pelo sklearn
from sklearn.preprocessing import Imputer

imputer =Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(previsores[:,0:3])
previsores[:,0:3]=imputer.transform(previsores[:,0:3])

#escalonamento 

from sklearn.preprocessing import StandardScaler

escala = StandardScaler()
previsores = escala.fit_transform(previsores)


