import os

use_gpu = True


def construct_layer_name(layer_id, layer_type, cell_type=''):
    return str(layer_id).zfill(2) + '_' + layer_type + (('' if cell_type == '' else '_') + cell_type)


def get_HH_mm_ss(td):
    days, seconds = td.days, td.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return hours, minutes, secs


def switch_backend(bk):
    os.environ['KMP_WARNINGS'] = '0'
    os.environ['KERAS_BACKEND'] = bk
    # import keras.backend as K
    # K.set_image_data_format("channels_last")
    if bk == 'tensorflow':
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
        import tensorflow as tf
        # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        print(f"tensorflow version: {tf.__version__}")

        # if use_gpu:
        #     # tf设置动态分配显存
        #     if str(tf.__version__)[0] == '1':
        #         from tensorflow.compat.v1 import GPUOptions
        #         from tensorflow.compat.v1 import ConfigProto
        #         from tensorflow.compat.v1 import Session
        #         from keras.backend.tensorflow_backend import set_session

        #         gpu_options = GPUOptions(allow_growth=True)
        #         set_session(Session(config=ConfigProto(gpu_options=gpu_options)))
        #     else:
        #         gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)

    elif bk == 'theano':
        os.environ['THEANO_FLAGS'] = "device=cuda,contexts=dev0->cuda0," \
                                    f"force_device=True,floatX=float32,lib.cnmem=1"
        import theano as th
        print(f"theano version: {th.__version__}")

    else:
        from cntk.device import try_set_default_device, gpu
        try_set_default_device(gpu(0))
        import cntk as ck
        print(f"cntk version: {ck.__version__}")


def get_layer_func(layer_type):
    import keras

    def _str2Str(s):
        res = ''
        i = 0
        while i < len(s):
            if i == 0:
                res += s[i].upper()
            elif s[i] == '_':
                i += 1
                res += s[i].upper()
            else:
                res += s[i]
            i += 1
        return res

    return keras.Input if layer_type == 'input_object' else getattr(keras.layers, _str2Str(layer_type))


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

# 以下为对上述layer的不同分类
# ------------------------------------------------------------------------

conv_layer_types = [
    'conv1D',
    'conv2D',
    'conv3D',
    'separable_conv1D',
    'separable_conv2D',
    'depthwise_conv2D',
    'conv2D_transpose',
    'conv3D_transpose',
]

pooling_layer_types = [
    'max_pooling1D',
    'max_pooling2D',
    'max_pooling3D',
    'average_pooling1D',
    'average_pooling2D',
    'average_pooling3D',
    # 'global_max_pooling1D',  # TODO
    # 'global_max_pooling2D',
    # 'global_max_pooling3D',
    # 'global_average_pooling1D',
    # 'global_average_pooling2D',
    # 'global_average_pooling3D',
]

recurrent_layer_types = [
    'time_distributed',
    'bidirectional',
    *RNN_layer_types
]

normalization_layers_types = [
    'batch_normalization',
]

reshape_layer_types = [
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
]

locally_connected_layer_types = [
    'locally_connected1D',
    'locally_connected2D',
]

normal_layer_types = seq_layer_types + RNN_layer_types + activation_layer_types
reduction_layer_types = pooling_layer_types + merging_layer_types

# TODO 进一步完善规则
activation_cond = (lambda **kwargs: kwargs['e1'] in ['dense', *conv_layer_types, *recurrent_layer_types])

dim_3_cond = (lambda **kwargs: kwargs.get('input_dim', None) == 3)
dim_4_cond = (lambda **kwargs: kwargs.get('input_dim', None) == 4)
dim_5_cond = (lambda **kwargs: kwargs.get('input_dim', None) == 5)

# layer的连接条件
layer_conditions = {
    **{layer_name: activation_cond for layer_name in activation_layer_types},
    'embedding': (lambda **kwargs: kwargs.get('e1', None) == 'input_object' and kwargs.get('input_dim', None) == 2),

    'conv1D': dim_3_cond,
    'conv2D': dim_4_cond,
    'conv3D': dim_5_cond,
    'separable_conv1D': dim_3_cond,
    'separable_conv2D': dim_4_cond,
    'depthwise_conv2D': dim_4_cond,
    'conv2D_transpose': dim_4_cond,
    'conv3D_transpose': dim_5_cond,

    'max_pooling1D': dim_3_cond,
    'max_pooling2D': dim_4_cond,
    'max_pooling3D': dim_5_cond,
    'average_pooling1D': dim_3_cond,
    'average_pooling2D': dim_4_cond,
    'average_pooling3D': dim_5_cond,
    'global_max_pooling1D': dim_3_cond,
    'global_max_pooling2D': dim_4_cond,
    'global_max_pooling3D': dim_5_cond,
    'global_average_pooling1D': dim_3_cond,
    'global_average_pooling2D': dim_4_cond,
    'global_average_pooling3D': dim_5_cond,

    'LSTM': dim_3_cond,
    'GRU': dim_3_cond,
    'simpleRNN': dim_3_cond,
    'time_distributed': (lambda **kwargs: kwargs.get('input_dim', None) >= 3),
    'bidirectional': (lambda **kwargs: kwargs.get('input_dim', None) == 3 or kwargs.get('input_dim', None) == 5),
    'convLSTM2D': dim_5_cond,

    'flatten': (lambda **kwargs: kwargs.get('input_dim', None) >= 3),
    'repeat_vector': (lambda **kwargs: kwargs.get('input_dim', None) == 2),
    'cropping1D': dim_3_cond,
    'cropping2D': dim_4_cond,
    'cropping3D': dim_5_cond,
    'up_sampling1D': dim_3_cond,
    'up_sampling2D': dim_4_cond,
    'up_sampling3D': dim_5_cond,
    'zero_padding1D': dim_3_cond,
    'zero_padding2D': dim_4_cond,
    'zero_padding3D': dim_5_cond,

    'locally_connected1D': dim_3_cond,
    'locally_connected2D': dim_4_cond,

    'subtract': (lambda **kwargs: kwargs.get('input_num', None) == 2),
    'dot': (lambda **kwargs: kwargs.get('input_num', None) == 2 and kwargs.get('output_shape', None) is None),
    'concatenate': (lambda **kwargs: kwargs.get('output_shape', None) is None)
}

losses = [
    'mean_squared_error',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'mean_squared_logarithmic_error',
    'squared_hinge',
    # 'hinge',  # 这个loss专用于二分类
    'categorical_hinge',
    'logcosh',
    'huber_loss',
    'categorical_crossentropy',
    # 'sparse_categorical_crossentropy',  #  这个loss会将输出值由one-hot变为平均值，即形状会改变
    'binary_crossentropy',
    'kullback_leibler_divergence',
    'poisson',
    'cosine_proximity',
]

optimizers = [
    'sgd',
    'rmsprop',
    'adagrad',
    'adadelta',
    'adam',
    'adamax',
    'nadam',
]

dataset_shape = {
    'cifar10': ((None, 32, 32, 3), (None, 10)),
    'mnist': ((None, 28, 28, 1), (None, 10)),
    'fashion_mnist': ((None, 28, 28, 1), (None, 10)),
    'imagenet': ((None, 224, 224, 3), (None, 1000)),
    'sinewave': ((None, 49, 1), (None, 1)),
    'price': ((None, 1, 240), (None,))
}
