import os

bk = 'cntk'
os.environ['KERAS_BACKEND'] = bk

import keras
import numpy as np
from keras import backend as K


model = keras.Sequential()
model.add(keras.layers.Dense(**{
        "units": 5,
        "activation": "linear",
        "use_bias": True,
        "kernel_initializer": "zeros",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
        "name": "06_dense"
    }, input_shape=(2, 1,)
))
model.add(keras.layers.ReLU(**{
    "name": "08_relu"
}))

model.get_layer('06_dense').set_weights(
    [np.array([[-6.5, -1.9,  4.5, -1.1, -2.8]], dtype=np.float32),
     np.array([4.0, 0.6, -4.5, -0.3, 1.6], dtype=np.float32)]
)

if bk == 'cntk':
    target_name = '08_relu'

    import cntk as C
    model_input = C.input_variable(model.get_layer(target_name).output_shape[1:], needs_gradient=True)
    tmp_input = keras.Input(tensor=model_input)

    layer_outputs = {}

    def get_output_of_layer(layer):
        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        if layer.name == target_name:
            return tmp_input

        prev_layers = [layer for node in layer._inbound_nodes for layer in node.inbound_layers]
        pl_outs = [get_output_of_layer(pl) for pl in prev_layers]

        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out

    tmp_output = get_output_of_layer(model.layers[-1])
    tmp_model = keras.models.Model(inputs=tmp_input, outputs=tmp_output)

    x = [np.array([[[-0., -0.,  0., -0., -0.],
                    [-0., -0.,  0., -0., -0.]]], dtype=np.float32)]
    y_true = [np.array([[[0., 0., 1., 0., 0.],
                         [1., 0., 0., 0., 0.]]], dtype=np.float32)]

    tmp_model.compile(loss='mean_absolute_percentage_error', optimizer='rmsprop')

    # ins = tmp_model._feed_inputs + tmp_model._feed_targets + tmp_model._feed_sample_weights
    # x, y, sample_weight = tmp_model._standardize_user_data(x, y_true)
    # ins_value = x + y + sample_weight

    ins = [model_input] + tmp_model._feed_targets + tmp_model._feed_sample_weights
    _, y, sample_weight = tmp_model._standardize_user_data([], y_true)
    ins_value = x + y + sample_weight

    grads = tmp_model.total_loss.grad({
        k: v
        for k, v in zip(ins, ins_value)
    }, [model_input])

    print(grads)


else:
    x = [np.array([[[1.],
                    [1.]]], dtype=np.float32)]
    y_true = [np.array([[[0., 0., 1., 0., 0.],
                         [1., 0., 0., 0., 0.]]], dtype=np.float32)]

    model.compile(loss='mean_absolute_percentage_error', optimizer='rmsprop')

    ins = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    x, y, sample_weight = model._standardize_user_data(x, y_true)
    ins_value = x + y + sample_weight

    get_gradients = K.function(
        ins + [K.learning_phase()],
        [model.total_loss]
        + model.outputs
        + [model.get_layer('06_dense').output]
        + model.optimizer.get_gradients(model.total_loss, model.outputs)
        + model.optimizer.get_gradients(model.total_loss, [model.get_layer('06_dense').output])
        + model.optimizer.get_gradients(model.total_loss, model.get_layer('06_dense').trainable_weights)
    )
    grads = get_gradients(ins_value + [1])

    print("loss: ", grads[0])
    print("output: ", grads[1])
    print("dense_output: ", grads[2])
    print("d(loss)/d(output): ", grads[3])
    print("d(loss)/d(dense_output): ", grads[4])
    grads = grads[5:]

# for p, g in zip(model.get_layer('06_dense').trainable_weights, grads):
#     print(f'\n{p.name}-grads:')
#     print(g)


# tensorflow
# loss:  20.0
# output:  [[[-0. -0.  0. -0. -0.]
#            [-0. -0.  0. -0. -0.]]]
# dense_output:  [[[-2.5       -1.3        0.        -1.4000001 -1.1999999]
#                  [-2.5       -1.3        0.        -1.4000001 -1.1999999]]]
# d(loss)/d(output):  [[[ -0.  -0. -10.  -0.  -0.]
#                       [-10.  -0.  -0.  -0.  -0.]]]
# d(loss)/d(dense_output):  [[[-0. -0. -0. -0. -0.]
#                             [-0. -0. -0. -0. -0.]]]

# 06_dense/kernel:0-grads:
# [[0. 0. 0. 0. 0.]]
# 06_dense/bias:0-grads:
# [0. 0. 0. 0. 0.]


# theano
# loss:  20.0
# output:  [[[0. 0. 0. 0. 0.]
#            [0. 0. 0. 0. 0.]]]
# dense_output:  [[[-2.5       -1.3        0.        -1.4000001 -1.1999999]
#                  [-2.5       -1.3        0.        -1.4000001 -1.1999999]]]
# d(loss)/d(output):  [[[ -0.  -0. -10.  -0.  -0.]
#                       [-10.  -0.  -0.  -0.  -0.]]]
# d(loss)/d(dense_output):  [[[ 0.  0. -5.  0.  0.]
#                             [ 0.  0. -0.  0.  0.]]]

# 06_dense/kernel-grads:
# [[ 0.  0. -5.  0.  0.]]
# 06_dense/bias-grads:
# [ 0.  0. -5.  0.  0.]


# cntk
# 06_dense/kernel-grads:
# [[0. 0. 0. 0. 0.]]

# 06_dense/bias-grads:
# [0. 0. 0. 0. 0.]
