# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:15:48 2018

@author: wstro
"""

import pandas as pd

base = pd.read_csv('census.csv')
classe= base.iloc[:,14].values
previsores = base.iloc[:,0:14].values

#alguns algoritmos nao aceitam previsores do tipo string e necessita a conversao
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_previsores=LabelEncoder()
#convertendo cada atributo categorico em numerico
#labels=label_previsores.fit_transform(previsores[:,1])
previsores[:,1]=label_previsores.fit_transform(previsores[:,1])
previsores[:,3]=label_previsores.fit_transform(previsores[:,3])
previsores[:,5]=label_previsores.fit_transform(previsores[:,5])
previsores[:,6]=label_previsores.fit_transform(previsores[:,6])
previsores[:,7]=label_previsores.fit_transform(previsores[:,7])
previsores[:,8]=label_previsores.fit_transform(previsores[:,8])
previsores[:,9]=label_previsores.fit_transform(previsores[:,9])
previsores[:,13]=label_previsores.fit_transform(previsores[:,13])

#pode existir inconsistencia ... pois valores nominais categoricos se transforma
#em ordinal ..
previsores[:,0]=label_previsores.fit_transform(previsores[:,0])

onehotencoder= OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelenconderclasse=LabelEncoder()

classe=labelenconderclasse.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
escalona = StandardScaler()
previsores=escalona.fit_transform(previsores)

