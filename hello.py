

import tensorflow as tf
print('hello')
print()

tf.compat.v1.disable_v2_behavior()

message = tf.constant('hello tensorflow')
session = tf.compat.v1.Session()
print(session.run(message))

print()
