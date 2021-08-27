import numpy as np
import tensorflow as tf

with tf.GradientTape() as tape:
    x = tf.Variable([[[0.6],
                      [0.6],
                      [0.3]]])

    max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=3)
    y = max_pool_1d(x)

g = tape.gradient(y, x)


print("gradients of max: ", g.numpy())

# gradients of max:  [[[1.]
#                      [0.]
#                      [0.]]]
