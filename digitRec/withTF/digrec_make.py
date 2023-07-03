import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
# Importing the MNIST dataset for handwritten digit recognition

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# Loading the training and testing data from the MNIST dataset
# x_train: training data (grayscale images)
# y_train: classification labels for training data

x_train=[1/256]*x_train
x_test=[1/256]*x_test
# Normalizing the input data by dividing each pixel value by 256

model = tf.keras.models.Sequential()
# Creating a sequential model using TensorFlow's Keras API

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Flattening the input data to a 1-dimensional array of size 784 (28x28)

model.add(tf.keras.layers.Dense(128, activation='relu'))
# Adding a dense layer with 128 units and ReLU activation function

model.add(tf.keras.layers.Dense(128, activation='relu'))
# Adding another dense layer with 128 units and ReLU activation function

model.add(tf.keras.layers.Dense(10, activation='softmax'))
# Adding a final dense layer with 10 units (corresponding to 10 digits) and softmax activation function

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Compiling the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric

model.fit(x_train, y_train, epochs=6)
# Training the model on the training data for 6 epochs

model.save('digrec.model')
# Saving the trained model to a file named 'deirec.model'
