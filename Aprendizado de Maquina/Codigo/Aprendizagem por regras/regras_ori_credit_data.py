# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:28:33 2018

@author: wstro
"""


import Orange
base = Orange.data.Table('credit-data.csv')
base.domain

base_divida= Orange.evaluation.testing.sample(base,n=0.25)
base_treinamento=base_divida[1]
base_teste=base_divida[0]

len(base_teste)
cn2_leaner=Orange.classification.rules.CN2Learner()

classificador= cn2_leaner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)
    
resultado= Orange.evaluation.testing.TestOnTestData(base_treinamento,base_teste,[classificador])

print(Orange.evaluation.CA(resultado))
