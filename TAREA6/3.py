import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Se agregó esto para cambiar una variable de entorno que estaba dando problemas

import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

class Polinomio(Layer):
    def __init__(self, **kwargs):
        super(Polinomio, self).__init__(**kwargs)
        self.a_0 = self.add_weight(initializer="random_normal", trainable=True) #Aqui le agregamos unos "pesos" que son los coeficientes del
        self.a_1 = self.add_weight(initializer="random_normal", trainable=True) #del polinomio. Le ponemos que son entrenables para que se pueda
        self.a_2 = self.add_weight(initializer="random_normal", trainable=True) #ajustar
        self.a_3 = self.add_weight(initializer="random_normal", trainable=True)

    def call(self, x):
        return self.a_0 + self.a_1 * x + self.a_2 * x**2 + self.a_3 * x**3 #Lo que regresa la capa es el polinomio evaluado con los 
    #coeficientes


model = keras.Sequential([
    Polinomio(), #hacemos nuestro modelo elcual tiene como unica capa la capa que hara el ajuste al polinomio
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

x = np.linspace(-1, 1, 100) #generamos un array que va de -1 a 1 y tiene 100 particiones
y_exact = np.cos(2 * x) #en ese array evaluamos la funcion que vamos a ajustar

history = model.fit(x, y_exact, epochs=100, verbose=0) #la decimos a la red cual es su input, cual es la funcion que quiere aproximae
#y por cuantas epocas va a correr

approx = model.predict(x) #guardamos en approx el resultado de la red


coeficientes = model.get_layer('polinomio').get_weights() #Recuperamos los coeficientes (pesos) que obtuvo la red
print(coeficientes)
a_0, a_1, a_2, a_3 = coeficientes #imprimimos los coeficientes
print("a_0:", a_0)
print("a_1:", a_1)
print("a_2:", a_2)
print("a_3:", a_3)

plt.plot(x, approx,  color='skyblue', label="approx") #graficamos la funcion aproximada y la funcion exacta
plt.plot(x, y_exact, color='orange', label="exact")
plt.legend()
plt.show()