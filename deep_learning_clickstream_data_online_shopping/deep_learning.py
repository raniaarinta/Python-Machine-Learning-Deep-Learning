import  pandas as pd
import pandas as pd
import numpy as np


# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

clickstream_train = pd.read_csv("clickstream+data+for+online+shopping.csv")
clickstream_train.apply(pd.to_numeric, errors='ignore')


print(clickstream_train.head())
clickstream_features  = clickstream_train .copy()
clickstream_labels =clickstream_features.pop('month')

clickstream_features = np.array(clickstream_features,dtype='float32')
clickstream_labels=np.array(clickstream_labels)
print(clickstream_features)
print(clickstream_labels)

normalize = preprocessing.Normalization()
norm_model = tf.keras.Sequential([
   #normalize,
  layers.Dense(64),
  layers.Dense(4)
])

norm_model.compile(loss = tf.losses.categorical_crossentropy,
                           optimizer = tf.optimizers.Adam(),
                   metrics=["accuracy"])

norm_model.fit(clickstream_features, clickstream_labels, epochs=5)