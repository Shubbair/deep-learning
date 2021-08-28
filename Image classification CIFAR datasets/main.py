"""
  *** Hussain Salih Mahdi ***
  _________Shubbair__________

TODO Image classification with CIFAR Datasets

Datasets info Link : https://www.cs.toronto.edu/~kriz/cifar.html

"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

y_train_categorical = keras.utils.to_categorical(
    y_train, num_classes=0, dtype='float32'
)
y_test_categorical = keras.utils.to_categorical(
    y_test, num_classes=0, dtype='float32'
)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(3000, activation='relu'),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_test_scaled, y_test_categorical, epochs=50)

print(np.argmax(model.predict(X_test_scaled)[0]))

print(classes[np.argmax(model.predict(X_test_scaled)[0])])
