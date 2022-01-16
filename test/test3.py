import numpy as np
import cntk as C

x = C.input_variable(shape=(1, 3, 1), needs_gradient=True)
x_val = np.array([[[0.6],
                   [0.6],
                   [0.3]]])

y = C.reduce_max(x)
g = y.grad({x: x_val})


print("gradients of max: ", g)

# gradients of max:  [[[[1.]
#                       [1.]
#                       [0.]]]]
