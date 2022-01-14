import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

layer = layers.Dense(3)
print(layer.weights)  # Empty


# Call layer on a test input
x = tf.ones((1, 4))
y = layer(x)
print("x : ", x)
print()
print(layer.weights)  # Now it has weights, of shape (4, 3) and (3,)
init = tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)

layer1 = layers.Dense(5, kernel_initializer=init, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")
layer1(x)
layer2(x)
layer3(x)
print("layer1.weights")
print(layer1.weights)
print("layer2.weights")
print(layer2.weights)
print("layer3.weights")
print(layer3.weights)
