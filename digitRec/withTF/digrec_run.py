import numpy as np
import tensorflow as tf


mnist = tf.keras.datasets.mnist
#this is the dataset import

(x_train, y_train),(x_test, y_test) = mnist.load_data()
#x_train is the training data(grayscale) and y_train is the classification label(correct)

x_train=[1/256]*x_train
x_test=[1/256]*x_test
#used a simple linear

model=tf.keras.models.load_model('digrec.model')

loss, accuracy  = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)
