# -*- coding: utf-8 -*-
"""
Created on Mon May 28 12:52:59 2018

@author: wstro
"""
contaDigitos = 0;
x= input("digite :")
valor=int(x);
print(len(x))
while valor!=0:
   
    valor = valor / 10;
    contaDigitos = contaDigitos + 1;

print(contaDigitos)