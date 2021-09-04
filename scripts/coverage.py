seq_layer_types = [
    'dense',
    # 'masking',
    'embedding',

    'conv1D',
    'conv2D',
    'conv3D',
    'separable_conv1D',
    'separable_conv2D',
    'depthwise_conv2D',
    'conv2D_transpose',
    'conv3D_transpose',

    'max_pooling1D',
    'max_pooling2D',
    'max_pooling3D',
    'average_pooling1D',
    'average_pooling2D',
    'average_pooling3D',
    'global_max_pooling1D',
    'global_max_pooling2D',
    'global_max_pooling3D',
    'global_average_pooling1D',
    'global_average_pooling2D',
    'global_average_pooling3D',

    'time_distributed',
    'bidirectional',

    'batch_normalization',

    'reshape',
    'flatten',
    'repeat_vector',
    'permute',
    'cropping1D',
    'cropping2D',
    'cropping3D',
    'up_sampling1D',
    'up_sampling2D',
    'up_sampling3D',
    'zero_padding1D',
    'zero_padding2D',
    'zero_padding3D',

    'locally_connected1D',
    'locally_connected2D',
]

RNN_layer_types = [
    'LSTM',
    'GRU',
    'simpleRNN',
    'convLSTM2D',
]

activation_layer_types = [
    'activation',

    'ReLU',
    'softmax',
    'leakyReLU',
    'PReLU',
    'ELU',
    'thresholded_ReLU',
]

merging_layer_types = [
    'concatenate',
    'average',
    'maximum',
    'minimum',
    'add',
    'subtract',
    'multiply',
    'dot',
]

# 所有layer类型
layer_types = seq_layer_types + RNN_layer_types + activation_layer_types + merging_layer_types
layer_types = [''.join(s.split('_')).lower() for s in layer_types]

import os
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import backend as K

def custom_objects():

    def no_activation(x):
        return x

    def leakyrelu(x):
        import keras.backend as K
        return K.relu(x, alpha=0.01)

    objects = {}
    objects['no_activation'] = no_activation
    objects['leakyrelu'] = leakyrelu
    return objects


selected_map = {name: 0 for name in layer_types}

for fn in Path(".").rglob("*.h5"):
    model = keras.models.load_model(str(fn), custom_objects=custom_objects())
    for layer in model.layers:
        type_name = layer.__class__.__name__.lower()
        if type_name in selected_map:
            selected_map[type_name] += 1
    K.clear_session()

cnt = 0
for selected in selected_map.values():
    if selected > 0:
        cnt += 1
coverage_rate = cnt / len(selected_map)

print(f'coverage: {coverage_rate}')
