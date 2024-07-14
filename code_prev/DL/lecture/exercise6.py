import numpy as np
import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input data
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

# Convert the labels to one-hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

z_dim = 32

# Define the generator model
model_encoder = Sequential([
    Input(shape=(784,)),
    tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(z_dim, kernel_size=(3,3), padding='same', activation='relu')
])

model_decoder = Sequential([
    Input(shape=(z_dim,)),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(784, activation='sigmoid')
])