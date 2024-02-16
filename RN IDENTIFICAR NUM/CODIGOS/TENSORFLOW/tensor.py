import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #Se agregó esto para cambiar una variable de entorno que estaba dando problemas
import tensorflow as tf
import keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
from keras import regularizers

epochs = 100   
batch_size = 60
learning_rate = 0.1
momentum = 0.9
n = 3

import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint

wandb.init(project="regularizadores")
wandb.config.learning_rate = learning_rate
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.momentum = momentum
###################
#import mlflow
#mlflow.tensorflow.autolog()
dataset=mnist.load_data()
#print(len(dataset))

(x_train, y_train), (x_test, y_test) = dataset
#print(y_train.shape)
#print(x_train.shape)
#print(x_test.shape)
#x_train=x_train[0:10000]
#x_test=x_train[0:10000]
#y_train=y_train[0:10000]
#y_test=y_train[0:10000]
x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
#print(x_trainv[3])
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')
x_trainv /= 255  # x_trainv = x_trainv/255
x_testv /= 255
#print("linea 40--------")
#print(x_trainv[3])
#print(x_train.shape)
#print(x_trainv.shape)
num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)
#print(y_trainc[6:15])
model = Sequential()
model.add(Dense(512, activation='sigmoid', input_shape=(784,)))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-6, l2=1e-5)))
#model.add(Dense(100, activation='softmax'))

model.add(Dense(num_classes, activation='sigmoid'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=SGD(learning_rate=learning_rate, momentum=momentum),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc),
                    callbacks=[
                        WandbMetricsLogger(log_freq="epoch"),
                        WandbModelCheckpoint("models")
                    ])




score = model.evaluate(x_testv, y_testc, verbose=1)
print(score)
a=model.predict(x_testv)
#b=model.predict_proba(x_testv)
print(a.shape)
print(a[n])
print("resultado correcto:")
print(y_testc[n])
#Para guardar el modelo en disco
model.save("C:\\Users\\jonat\\git\\RN IDENTIFICAR NUM\\CODIGOS\\TENSORFLOW\\red.keras")
exit()
#para cargar la red:
modelo_cargado = tf.keras.models.load_model("C:\\Users\\jonat\\git\\RN IDENTIFICAR NUM\\CODIGOS\\TENSORFLOW\\red.keras")