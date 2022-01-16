import numpy as np
import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable([[[0.6],
                      [0.6],
                      [0.3]]])

    y = tf.reduce_max(x)
g = tape.gradient(y, x)


print("gradients of max: ", g.numpy())

# gradients of max:  [[[0.5]
#                      [0.5]
#                      [0. ]]]
