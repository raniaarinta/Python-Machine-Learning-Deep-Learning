import  tensorflow as tf
import keras
import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split

data=np.loadtxt("heart_failure_clinical_records_dataset.csv",delimiter=',',skiprows=1)
X=data[:, :-1]
Y= data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

model= keras.Sequential()
model.add(keras.layers.Dense(15, input_dim=len(X[0,:]),activation=tf.nn.relu))
model.add(keras.layers.Dense(20,activation=tf.nn.relu))
model.add(keras.layers.Dense(25,activation=tf.nn.relu))
model.add(keras.layers.Dense(20,activation=tf.nn.relu))
model.add(keras.layers.Dense(1,activation=tf.nn.sigmoid))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, epochs=50, batch_size=20)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
print('Loss:', test_loss)