# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:15:04 2018

@author: wstro
"""

import Orange
base = Orange.data.Table('risco-credito.csv')
base.domain

cn2_leaner=Orange.classification.rules.CN2Learner()

classificador= cn2_leaner(base)

for regras in classificador.rule_list:
    print(regras)
    
resultado= classificador([['boa','alta','nenhuma','acima_35'],['ruim','alta','adequada','0_15']])

for i in resultado:
    print(base.domain.class_var.values[i])