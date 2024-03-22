import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Se agregó esto para cambiar una variable de entorno que estaba dando problemas

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np

class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        min = tf.cast(tf.reduce_min(data),tf.float32)
        max = tf.cast(tf.reduce_max(data),tf.float32)
        x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x) #Le indicamos la variable sobre la cual va a derivar
                y_pred = self(x, training=True) #Nos da el resultado de pasar los inputs por la red
            dy = tape2.gradient(y_pred, x) #derivada de la prediccion del modelo respecto a x
            x_o = tf.zeros((batch_size,1)) #tensor de puros ceros para que sea el input para la condicion inicial
            y_o = self(x_o,training=True) #valor del modelo en x=0 (condicion inicial)
            eq =  x * dy + y_pred - x * x * tf.cos(x) #Es la ecuacion diferencial pero es paso todo al lado izquierdo de la ecuacion.
             #Esto queremos que se aproxime a cero
            ic = 0. #valor que queremos para la condicion inicial o el modelo en x_0
            loss = self.mse(0., eq) + self.mse(y_o,ic) #queremos que loss sea muy pequeño y loss es la suma del error que tenemos en la
             #ecuacion como tal y el error en el valor inicial

  
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables)) # Se actualizan los pesos

            self.loss_tracker.update_state(loss) #Actualiza el valor de la funcion de costo

            return {"loss": self.loss_tracker.result()} #Nos regresa elvalor de la funcion de costo


model = ODEsolver()


model.add(Dense(200,activation='tanh', input_shape=(1,)))
model.add(Dense(200,activation='tanh'))
model.add(Dense(150,activation='relu'))
model.add(Dense(1))


model.summary()

model.compile(optimizer=RMSprop(),metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=1000,verbose=0) #entrenamos el modelo


x_testv = tf.linspace(-5,5,100)
a=model.predict(x_testv)
plt.plot(x_testv,a, color='skyblue', label="aprox")
plt.plot(x_testv,x*np.sin(x) - (2/x)*(np.sin(x) - x*np.cos(x)),color='orange', label="exact")
plt.legend()
plt.show()