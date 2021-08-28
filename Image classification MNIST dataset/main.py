"""
  *** Hussain Salih Mahdi ***
  _________Shubbair__________

TODO Image classification with MNIST Datasets

"""

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images_scaled = train_images / 255.0
test_images_scaled = test_images / 255.0

plt.imshow(train_images_scaled[0])

model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(784,activation='relu'),
      keras.layers.Dense(100,activation='relu'),
      keras.layers.Dense(10,activation='sigmoid'),

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(test_images_scaled, test_labels, epochs=10)

print(np.argmax(model.predict(test_images_scaled)[0]))

print(class_names[np.argmax(model.predict(test_images_scaled)[0])])

plt.matshow(train_images[np.argmax(model.predict(test_images_scaled)[0])])
plt.show()
