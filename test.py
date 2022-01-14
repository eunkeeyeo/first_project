import tensorflow as tf
print(tf.executing_eagerly())

import time


tm = time.localtime()
print(tm.tm_hour)
print(tm.tm_mon)
print(tm.tm_sec)

print("{}:{}:{}".format(tm.tm_hour, tm.tm_min, tm.tm_sec))
print()
now = "{}:{}:{}".format(tm.tm_hour, tm.tm_min, tm.tm_sec)
print("what's up {}".format(now))



# 신경망 훈련
print("테스트으")
start = time.time()-2
runtime = time.time() - start
per =  (1-(runtime / 66.2))
print("Learning time : %.1f (%.1f faster)" % (runtime ,per))

