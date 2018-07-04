# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:51:12 2018

@author: wstro
"""

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer,SigmoidLayer,BiasUnit
from pybrain.structure import FullConnection

rede=FeedForwardNetwork()

camada_de_entrada=LinearLayer(2)
camada_oculta=SigmoidLayer(3)
camada_saida= SigmoidLayer(1)

bias_1=BiasUnit()
bias_2=BiasUnit()

rede.addModule(camada_de_entrada)
rede.addModule(camada_oculta)
rede.addModule(camada_saida)
rede.addModule(bias_1)
rede.addModule(bias_2)

entradaOculta=FullConnection(camada_de_entrada,camada_oculta)
Oculta_a_saida=FullConnection(camada_oculta,camada_saida)

biasOculta=FullConnection(bias_1,camada_oculta)
biasSaida=FullConnection(bias_2,camada_saida)

rede.sortModules()

print(rede)
print(entradaOculta.params)
print(Oculta_a_saida.params)
print(biasOculta.params)
print(biasSaida.params)