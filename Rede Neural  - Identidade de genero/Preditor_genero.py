# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:17:39 2018

@author: wstro
"""

import pandas as pd
import pickle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import tempfile

import re

#Classificadores
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid

#########################################################

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict,train_test_split
from sklearn.metrics import confusion_matrix,precision_score,f1_score
import itertools

Forest = pickle.load(open('ramdomForest_identidadeGenero.sav', 'rb'))
probabilidades = Forest.predict_proba(vetorizador.transform([""]))[0]
print("Feminino: {}% de chance | Masculino: {}% de chance.".format(probabilidades[0]*100, probabilidades[1]*100))