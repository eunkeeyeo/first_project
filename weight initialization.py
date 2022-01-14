# 가중치 초기화에 따른 mnist 손글씨 학습 정확도.
# https://blog.naver.com/teach3450/221744273710

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
tf.compat.v1.disable_v2_behavior()
#
# tf.enable_eager_execution()  # eager 모드

""" mnist를 불러오는 함수 """


def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    train_data = np.expand_dims(train_data, axis=-1)  # [N, 28, 28] -> [N, 28, 28, 1]

    test_data = np.expand_dims(test_data, axis=-1)  # [N, 28, 28] -> [N, 28, 28, 1]

    train_data, test_data = normalize(train_data, test_data)  # 정규화

    """ 숫자의 10개 레이블 """

    train_labels = to_categorical(train_labels, 10)  # [N,] -> [N, 10]

    test_labels = to_categorical(test_labels, 10)  # [N,] -> [N, 10]

    return train_data, train_labels, test_data, test_labels


""" 정규화 함수 선언 """


def normalize(train_data, test_data):
    train_data = train_data.astype(np.float32) / 255.0

    test_data = test_data.astype(np.float32) / 255.0

    return train_data, test_data


""" Performance function """


def loss_fn(model, images, labels):
    logits = model(images, training=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return loss


def accuracy_fn(model, images, labels):
    logits = model(images, training=False)

    prediction = tf.equal(tf.argmax(logits, -1), tf.argmax(labels, -1))

    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy


def grad(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)

    return tape.gradient(loss, model.variables)


""" 모델을 구성할 함수 선언 """


def flatten():
    return tf.keras.layers.Flatten()  # 평탄화 과정


def dense(label_dim, weight_init):
    return tf.keras.layers.Dense(units=label_dim, use_bias=True, kernel_initializer=weight_init)


def relu():
    return tf.keras.layers.Activation(tf.keras.activations.relu)


""" 모델 생성 """


def create_model_function(label_dim):
    weight_init = tf.keras.initializers.glorot_uniform()

    model = tf.keras.Sequential()

    model.add(flatten())

    for i in range(2):
        model.add(dense(256, weight_init))

        model.add(relu())

    model.add(dense(label_dim, weight_init))

    return model


""" dataset """

train_x, train_y, test_x, test_y = load_mnist()

""" parameters """

learning_rate = 0.001  # 학습률

batch_size = 128  # 미니 배치 크기

training_epochs = 1

training_iterations = len(train_x) // batch_size

label_dim = 10  # 숫자 레이블 개수

""" Graph Input using Dataset API """

""" shuffle의 buffer_size는 train:60000, test:10000 보다 크게 """

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)). \
 \
    shuffle(buffer_size=100000). \
 \
    prefetch(buffer_size=batch_size). \
 \
    batch(batch_size). \
 \
    repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)). \
 \
    shuffle(buffer_size=100000). \
 \
    prefetch(buffer_size=len(test_x)). \
 \
    batch(len(test_x)). \
 \
    repeat()

""" Dataset Iterator """

train_iterator = train_dataset.make_one_shot_iterator()

test_iterator = test_dataset.make_one_shot_iterator()

""" Model """

network = create_model_function(label_dim)

""" Training """

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

checkpoint = tf.train.Checkpoint(dnn=network)

global_step = tf.train.create_global_step()

for epoch in range(training_epochs):

    for idx in range(training_iterations):
        train_input, train_label = train_iterator.get_next()

        grads = grad(network, train_input, train_label)

        optimizer.apply_gradients(grads_and_vars=zip(grads, network.variables), global_step=global_step)

        train_loss = loss_fn(network, train_input, train_label)

        train_accuracy = accuracy_fn(network, train_input, train_label)

        test_input, test_label = test_iterator.get_next()

        test_accuracy = accuracy_fn(network, test_input, test_label)

        print(

            "Epoch: [%2d] [%5d/%5d], train_loss: %.8f, train_accuracy: %.4f, test_Accuracy: %.4f" \
 \
            % (epoch, idx, training_iterations, train_loss, train_accuracy, test_accuracy))