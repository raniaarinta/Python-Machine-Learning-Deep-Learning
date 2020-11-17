import tensorflow as tf
import  keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sign_language_train = pd.read_csv("sign_mnist_train.csv")
sign_language_test=pd.read_csv("sign_mnist_test.csv")

Y_train = sign_language_train["label"]


X_train = sign_language_train.drop(labels = ["label"],axis = 1)

Y_test = sign_language_test["label"]
X_test = sign_language_test.drop(labels = ["label"],axis = 1)


X_train = np.array(X_train, dtype='float32')
X_test = np.array(X_test, dtype='float32')
Y_train = np.array(Y_train, dtype='float32')
Y_test = np.array(Y_test, dtype='float32')


X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)

_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = keras.utils.np_utils.to_categorical(Y_train, num_classes = 25)
Y_test = keras.utils.np_utils.to_categorical(Y_test, num_classes = 25)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)

model= keras.Sequential()
model.add(keras.layers.Dense(128,activation='relu', input_shape = (28,28,1)))
model.add(keras.layers.Dense(128,activation='relu'))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation='relu'))
model.add(keras.layers.Dense(25,activation='softmax'))

model.compile(optimizer="adam" , loss = "categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=10)
