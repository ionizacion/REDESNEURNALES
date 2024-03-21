import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Se agregó esto para cambiar una variable de entorno que estaba dando problemas

import tensorflow as tf
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Input
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

class ImagenAGrises(Layer):
    def __init__(self, **kwargs):
        super(ImagenAGrises, self).__init__(**kwargs)

    def call(self, inputs):
        imagengrises = tf.image.rgb_to_grayscale(inputs) #Le ponemos que queremos que nos regrese la capa
        return imagengrises


model = Sequential()
model.add(Input(shape=(None, None, 3)) ) #Se le pone que el input es "None" porque no conocemos de que tamaño es la imagen. El 3 representa
#que esta en tres canales (RGB) el input
model.add(ImagenAGrises()) #Agregamos al modelo la capa personalizada


model.compile() #Compilamos el modelo


imagen_a_color = np.random.random((1, 64, 64, 3))  #Aqui generamos una imagen de 64 por 64 pixeles donde el valor de cada canal de cada pixel
#es un valor aleatorio
plt.imshow(imagen_a_color[0]) 
plt.show() #Mostramos la imagen

tf.keras.preprocessing.image.save_img("imagen_a_color.jpg", imagen_a_color[0]) #Guardamos la imagen a color




imagen_esc_grises = model.predict(imagen_a_color) #Introducimos la imagen a color al modelo para que nos regrese la imgen a escala de grises



tf.keras.preprocessing.image.save_img("imagen_esc_grises.jpg", imagen_esc_grises[0]) #Guardamos la imagen a escala de grises

plt.imshow(imagen_esc_grises[0], cmap = plt.cm.gray) #Mostramos la imagen en escala de grises. Aqui se le tiene que especificar a mathplotlib
#que queremos que este en escala de grises. Como la imagen ahora solo tiene un canal mathplotlib solo lo interpreta como una especie de 
#intensidad y tu puedes elegir en que escala de colores quieres tu imagen, por eso mismo más adelante muestro la imagenes que se guardaron anteriormente

plt.show()

image = Image.open("imagen_a_color.jpg") #Mostramos la imagen a color que se guardo
image.show()


image = Image.open("imagen_esc_grises.jpg") #Mostramos la imagen a escala de grises que se guardo
image.show()
