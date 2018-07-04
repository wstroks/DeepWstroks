# -*- coding: utf-8 -*-
"""
Created on Wed May 30 00:10:56 2018

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
import keras
from keras.models import Sequential
from keras.layers import Dense
classificador=Sequential()
classificador.add(Dense(units=2,activation='relu',input_dim=3))
classificador.add(Dense(units=2,activation='relu'))
classificador.add(Dense(units=1,activation='sigmoid'))#camada binária retorna sigmode

classificador.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classificador.fit(previsores_treinamento,classe_treinamento,batch_size=10,nb_epoch=100)

previsoes=classificador.predict(previsores_teste)
previsoes=(previsoes>0.5)

from sklearn.metrics import confusion_matrix,accuracy_score

precisao=accuracy_score(classe_teste,previsoes)

matriz=confusion_matrix(classe_teste,previsoes)
