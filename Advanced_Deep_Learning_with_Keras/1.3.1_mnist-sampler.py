import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import tensorflow as tf



(x_train, y_train) , (x_test, y_test) = mnist.load_data()
unique, counts = np.unique(y_train, return_counts=True)
print("Train labels: ", dict(zip(unique, counts)))

unique, counts = np.unique(y_test, return_counts=True)
print("Test Labels: ", dict(zip(unique, counts)))

indexes = np.random.randint(0, x_train.shape[0], size = 25)
images = x_train[indexes]
labels = y_train[indexes]

plt.figure(figsize=(5,5))
for i in range(len(indexes)):
    plt.subplot(5, 5, i+1)
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')




plt.show()

my_path = os.path.abspath(r'C:\Users\eunkeeyeo\Desktop\workspace\Advanced_Deep_Learning_with_Keras')

plt.savefig(my_path + r"mnist_samples.png")
plt.close('all')

