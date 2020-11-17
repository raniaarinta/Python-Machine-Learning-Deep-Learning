import  tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv("iris.csv")
df = pd.DataFrame(data)
iris = df.values
X = iris[1:, 1:5].astype(float)
Y = iris[1:, 5]
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
y_dummy = keras.utils.np_utils.to_categorical(encoded_Y)

model= keras.Sequential()
model.add(keras.layers.Dense(9, input_dim=4,activation=tf.nn.relu))
model.add(keras.layers.Dense(3,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y_dummy, epochs=10, batch_size=1)
model.summary()
