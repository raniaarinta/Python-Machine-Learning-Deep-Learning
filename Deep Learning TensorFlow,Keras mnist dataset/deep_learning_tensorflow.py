import tensorflow as tf
import keras as k
import numpy as np
import matplotlib.pyplot as plt
mnist= k.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#normalize the data
x_train=k.utils.normalize(x_train)
x_test=k.utils.normalize(x_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4)
model.save('mnist.model')
predict_model= tf.keras.models.load_model('mnist.model')
p=predict_model.predict(x_test)
print(np.argmax(p[0]))
plt.imshow((x_test[0]))
plt.show()