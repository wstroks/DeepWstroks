# Etapa 1: Importação das bibliotecas
import os
import requests
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify

print(tf.__version__)

# Etapa 2: Carregamento do modelo pré-treinado
with open("fashion_model_flask.json", "r") as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
model.load_weights("fashion_model_flask.h5")
model.summary()

# Etapa 3: Criação da API em Flask
app = Flask(__name__)

# Função para classificação de imagens
@app.route("/<string:img_name>", methods = ["POST"])
def classify_image(img_name):
    upload_dir = "uploads/"
    image = imread(upload_dir + img_name)
    
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    # [1, 28, 28] -> [1, 784]
    prediction = model.predict([image.reshape(1, 28 * 28)])
    
    return jsonify({"object_identified": classes[np.argmax(prediction[0])]})

@app.route("/")
def home():
   return("Hello World2")
if __name__ == "__main__":
   app.run()