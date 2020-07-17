import tensorflow as tf
import keras
import numpy as  np
import matplotlib.pyplot as plt
mnist =tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

plt.imshow(x_train[0],cmap=plt.cm.binary)
#plt.show()

#normalization
x_train= tf.keras.utils.normalize(x_train,axis=1)
x_test= tf.keras.utils.normalize(x_test,axis=1)

#build model
model= tf.keras.models.Sequential()
#input layer
model.add(tf.keras.layers.Flatten())
#1 hidden layer 100 neurons activation fuction using relu
model.add(tf.keras.layers.Dense(100,activation=tf.nn.relu))

#outputlayer with 10 number of classfication and activation fuction using softmax
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

#paramater for the training model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train model
model.fit(x_train, y_train, epochs=3)




