import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN
from tensorflow.keras.utils import to_categorical, plot_model
from keras.datasets import mnist
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



# 이미지 차원(정사각형으로 가정)
image_size = x_train.shape[1]

# 크기 재조정 및 정규화
x_train = np.reshape(x_train, [-1, image_size, image_size])
x_test = np.reshape(x_test, [-1, image_size, image_size])
x_train = x_train.astype('float32') /255
x_test = x_test.astype('float32') /255

# 신경망 매개변수
input_shape = (image_size, image_size)
batch_size = 128
units = 512
dropout = 0.45

# init = tf.keras.initializers.glorot_uniform(seed=None)
# init = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
init = tf.keras.initializers.he_uniform(seed=None)

activ = 'sigmoid'


# 모델 : 256개 유닛으로 구성된 RNN. 입력 : 28시간 단계, 28 항목으로 구성된 벡터
model = Sequential()
model.add(SimpleRNN(units=units, dropout=dropout, input_shape=input_shape, activation=activ, kernel_initializer=init))
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file='rnn_mnist {}.png'.format(now), show_shapes = True)





# 원 핫 벡터의 손실 함수
# sgd 최적화 사용
# 분류 작업의 지표로 정확도(accuracy) 를 사용 것이 저합함
print("원 핫 벡터의 손실 함수")
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 신경망 훈련
print("신경망 훈련")
start = time.time()
model.fit(x_train, y_train, epochs = 10, batch_size = batch_size) # fit : 훈련을 시작

# 일반화가 제대로 됐는지 확인하기 위해 테스트 데이터세트로 모델 검증
print("일반화가 제대로 됐는지 확인하기 위해 테스트 데이터세트로 모델 검증")
loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("MNIST - RNN Test", now)
print("MEMO          : dropout = 0.45 ")
print("Test accuracy : %.2f%%" % (100.0*acc))
print("Learning time : %.2fs" % (time.time() - start))


runtime = time.time() - start
per =  (1-(runtime / 66.2))
# print("Learning time : %.1f (%.1f faster)" % (runtime ,per))