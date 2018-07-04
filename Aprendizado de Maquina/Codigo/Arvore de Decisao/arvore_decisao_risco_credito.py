# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:23:54 2018

@author: wstro
"""


import pandas as pd

base = pd.read_csv('risco-credito.csv')

preditores=base.iloc[:,0:4].values
classe=base.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder

label= LabelEncoder()
preditores[:,0]=label.fit_transform(preditores[:,0])
preditores[:,1]=label.fit_transform(preditores[:,1])
preditores[:,2]=label.fit_transform(preditores[:,2])
preditores[:,3]=label.fit_transform(preditores[:,3])


from sklearn.tree import DecisionTreeClassifier,export

classificador= DecisionTreeClassifier(criterion='entropy')
classificador.fit(preditores,classe)
print(classificador.feature_importances_)

export.export_graphviz(classificador,
                       out_file='arvore.dot',
                       feature_names=['historia','divida','garantias','renda'],
                       class_names=['alto','moderado','baixo'],
                       filled=True,
                       leaves_parallel=True)
#historia boa,divida alta,garantias nenhuma,renda>15
#historia ruim,divida alta,garantias adequada,randa<15

resultado=classificador.predict([[0,0,1,2],[3,0,0,0]])


