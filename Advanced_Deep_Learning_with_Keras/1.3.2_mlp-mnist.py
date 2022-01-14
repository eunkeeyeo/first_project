import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import plot_model
# from keras.utils.vis_utils import plot_model

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import time
tm = time.localtime()
now = "{} {} {}".format(tm.tm_hour, tm.tm_min, tm.tm_sec)
tf.compat.v1.disable_v2_behavior()



# MNIST 데이터 세트 로딩
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

# 레이블 개수 계산
num_labels = len(np.unique(y_train)) # np.unique : element의 종류들을 모아줌.  np.unique(y_train) :  [0 1 2 3 4 5 6 7 8 9]  ,   num_labels = 10


# 원-핫 벡터로 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# y-train before :  [5 0 4 ... 5 6 8]
# y-train after :  [[0. 0. 0. ... 0. 0. 0.]
#  [1. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 0. 0.]
#  [0. 0. 0. ... 0. 1. 0.]]



# 이미지 차원(정사각형으로 가정)
image_size = x_train.shape[1]
input_size = image_size * image_size # image_size :  28  ,  input_size :  784

# 크기 조정, 정규화(Normalization)
x_train = np.reshape(x_train, [-1, input_size]) # [-1, input_size] : "-1"을 넣어서 행이 몇개가 되든, 열이input size만큼 되게 하는 trick
x_train = x_train.astype('float32') /255 # 데이터의 0~255 을 0~1로 normalization 을 위해서 x_train의 자료형을 'int' -> 'float32'로 변경
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# 신경망 매개변수
batch_size = 128
hidden_units = 256 # 유닛의 수. 함수의 복잡도를 나타내는 지표임.
dropout = 0.45

init = tf.keras.initializers.glorot_uniform(seed=None)
# init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
# init = tf.keras.initializers.he_uniform(seed=None)


# 모델 : 3개의 계층으로 이루어진 MLP(각 계층 다음에는 ReLU와 드롭아웃을 적용)
# print("모델 : 3개의 계층으로 이루어진 MLP(각 계층 다음에는 ReLU와 드롭아웃을 적용)")
model = Sequential()

model.add(Dense(hidden_units, input_dim = input_size, kernel_initializer=init)) # Dense = Densely Conneted Layer = MLP(Multilayer Perceptron) = 다층퍼셉트론
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units, kernel_initializer=init))
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(num_labels, kernel_initializer=init))


# 원-핫 벡터 출력0
# print("원-핫 벡터 출력")
model.add(Activation('softmax'))
model.summary()
# print("원 핫 벡터의 손실 함수")
plot_model(model, to_file='mpl-mnist {}.png'.format(now), show_shapes = True)


# 원 핫 벡터의 손실 함수
# adam 최적화 사용
# 분류 작업의 지표로 정확도(accuracy) 를 사용 것이 저합함
print("원 핫 벡터의 손실 함수")
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 신경망 훈련
print("신경망 훈련")
start = time.time()
model.fit(x_train, y_train, epochs = 20, batch_size = batch_size) # fit : 훈련을 시작


# 일반화가 제대로 좼는지 확인하기 위해 테스트 데이터세트로 모델 검증
# print("일반화가 제대로 됐는지 확인하기 위해 테스트 데이터세트로 모델 검증")
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("MNIST - MLP Test", now)
print("MEMO          : kernel init test ")
print("Test accuracy : %.2f%%" % (100.0*acc))
print("Learning time : %.1fs" % (time.time() - start))




# in case of "hidden_unit = 256" : Test accuracy : 98.4% / 15us/sample
# in case of "hidden_unit = 128" : Test accuracy : 97.7% / 11us/sample
# in case of "hidden_unit = 512" : Test accuracy : 98.3% / 40us/sample
# in case of "hidden_unit = 1024" : Test accuracy : 98.4% / 107us/sample