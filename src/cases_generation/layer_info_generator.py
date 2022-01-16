from typing import Tuple, Optional

from .variable_generator import VariableGenerator
from .output_shape_calculator import OutputShapeCalculator
from utils.selection import Roulette
from utils.utils import seq_layer_types, RNN_layer_types, activation_layer_types, merging_layer_types


class LayerInfoGenerator(object):
    '''layer信息生成器
    '''

    def __init__(self, variable_generator: VariableGenerator, selector: Roulette):
        super().__init__()
        self.layer_infos = LayerInfo(variable_generator, self)

        # 可选层
        from utils.utils import layer_types
        self.__layer_funcs = {name: getattr(self.layer_infos, name + '_layer') for name in layer_types}

        # Roulette处理器
        self.__selector = selector

    def generate(self, input_shape: Tuple[Optional[int]], last_layer: Optional[str] = None, pool: Optional[list] = None):
        '''随机返回一个layer的name、参数字典和理论输出形状
        '''
        input_dim = len(input_shape)
        normal_pool = set(seq_layer_types+RNN_layer_types+activation_layer_types)
        pool = list(set(pool) & normal_pool) if pool is not None else list(normal_pool)
        element = self.__selector.choose_element(pool=pool,
                                                 e1=last_layer,
                                                 input_dim=input_dim)
        if element is None:  # 没有符合的层类型
            return None, None, input_shape
        return self.__layer_funcs[element](input_shape=input_shape)

    def generate_merging_layer(self, input_num: int = 2, pool: Optional[list] = None, output_shape: Optional[tuple] = None):
        '''随机返回一个merging layer的name、参数字典、期望输入形状列表、理论输出形状
        '''
        pool = list(set(pool) & set(merging_layer_types)) if pool is not None else merging_layer_types
        return self.__layer_funcs[self.__selector.choose_element(pool=pool,
                                                                 input_num=input_num,
                                                                 output_shape=output_shape)](input_num=input_num, output_shape=output_shape)

    def generate_RNN_layer(self, input_shape: Tuple[Optional[int]], pool: Optional[list] = None):
        '''RNN类的layer(用于bidirectional)
        '''
        input_dim = len(input_shape)
        pool = list(set(pool) & set(RNN_layer_types)) if pool is not None else RNN_layer_types
        return self.__layer_funcs[self.__selector.choose_element(pool=pool,
                                                                 input_dim=input_dim)](input_shape=input_shape)


class LayerInfo(object):

    def __init__(self, variable_generator: VariableGenerator, generator):
        super().__init__()
        self.__random = variable_generator
        self.__output_shape = OutputShapeCalculator()
        self.__generator = generator

    def dense_layer(self, input_shape: Tuple[Optional[int]], units: Optional[int] = None):
        if units is None:
            units = self.__random.ele_size()
        args = dict(
            units=units,
            activation='linear',  # self.__random.activation_func() 将dense和activation层分离
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'dense', args, self.__output_shape.dense_layer(input_shape, **args)

    def activation_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(activation=self.__random.activation_func())
        return 'activation', args, self.__output_shape.activation_layer(input_shape)

    def embedding_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入为2D向量
        '''
        args = dict(
            input_dim=self.__random.vocabulary_size(),
            output_dim=self.__random.ele_size(),
            embeddings_initializer='random_uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,  # 暂不支持mask
            input_length=None,
        )
        return 'embedding', args, self.__output_shape.embedding_layer(input_shape, **args)

    def masking_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(mask_value=self.__random.val_size())
        return 'masking', args, self.__output_shape.masking_layer(input_shape)

    def conv1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        padding = self.__random.choice(["valid", "same", "causal"])
        is_channels_last = True if padding == "causal" else self.__random.boolean()
        kernel_size, strides, data_format, dilation_rate = self.__random.conv_args(input_shape, dim_num=1, is_channels_last=is_channels_last)
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'conv1D', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=1, **args)

    def conv2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        kernel_size, strides, data_format, dilation_rate = self.__random.conv_args(input_shape, dim_num=2)
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=kernel_size,
            strides=strides,
            padding=self.__random.choice(["valid", "same"]),
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'conv2D', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=2, **args)

    def conv3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5+D向量
        '''
        kernel_size, strides, data_format, dilation_rate = self.__random.conv_args(input_shape, dim_num=3)
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=kernel_size,
            strides=strides,
            padding=self.__random.choice(["valid", "same"]),
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'conv3D', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=3, **args)

    def separable_conv1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        kernel_size, strides, data_format, dilation_rate = self.__random.conv_args(input_shape, dim_num=1)
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=kernel_size,
            strides=strides,
            padding=self.__random.choice(["valid", "same"]),
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=self.__random.ele_size(),
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            depthwise_initializer='random_uniform',
            pointwise_initializer='random_uniform',
            bias_initializer='random_uniform',
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            pointwise_constraint=None,
            bias_constraint=None,
        )
        return 'separable_conv1D', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=1, **args)

    def separable_conv2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        kernel_size, strides, data_format, dilation_rate = self.__random.conv_args(input_shape, dim_num=2)
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=kernel_size,
            strides=strides,
            padding=self.__random.choice(["valid", "same"]),
            data_format=data_format,
            dilation_rate=dilation_rate,
            depth_multiplier=self.__random.ele_size(),
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            depthwise_initializer='random_uniform',
            pointwise_initializer='random_uniform',
            bias_initializer='random_uniform',
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            pointwise_constraint=None,
            bias_constraint=None,
        )
        return 'separable_conv2D', args, self.__output_shape.conv_layer(input_shape=input_shape, dim_num=2, **args)

    def depthwise_conv2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:3] if is_channels_last else input_shape[2:4]
        _strides = self.__random.sizes_with_limitation(window_limitation)
        v = min(_strides)
        args = dict(
            kernel_size=self.__random.kernel_size(window_limitation),
            strides=[v] * len(_strides),
            padding=self.__random.choice(["valid", "same"]),
            depth_multiplier=self.__random.ele_size(),
            data_format="channels_last" if is_channels_last else "channels_first",
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            depthwise_initializer='random_uniform',
            bias_initializer='random_uniform',
            depthwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
        )
        return 'depthwise_conv2D', args, self.__output_shape.depthwise_conv2D_layer(input_shape=input_shape, **args)

    def conv2D_transpose_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:3] if is_channels_last else input_shape[2:4]
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation),
            padding=self.__random.choice(["valid", "same"]),
            data_format="channels_last" if is_channels_last else "channels_first",
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'conv2D_transpose', args, self.__output_shape.conv_transpose_layer(input_shape=input_shape, dim_num=2, **args)

    def conv3D_transpose_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:4] if is_channels_last else input_shape[2:5]
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation),
            padding=self.__random.choice(["valid", "same"]),
            data_format="channels_last" if is_channels_last else "channels_first",
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'conv3D_transpose', args, self.__output_shape.conv_transpose_layer(input_shape=input_shape, dim_num=3, **args)

    def max_pooling1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            pool_size=self.__random.kernel_size(input_shape[1:2])[0],
            strides=self.__random.sizes_with_limitation(input_shape[1:2])[0] if self.__random.boolean() else None,
            padding=self.__random.choice(["valid", "same"]),
        )
        return 'max_pooling1D', args, self.__output_shape.pooling1D_layer(input_shape=input_shape, **args)

    def max_pooling2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:3] if is_channels_last else input_shape[2:4]
        args = dict(
            pool_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else None,
            padding=self.__random.choice(["valid", "same"]),
            data_format="channels_last" if is_channels_last else "channels_first",
        )
        return 'max_pooling2D', args, self.__output_shape.pooling_layer(input_shape=input_shape, dim_num=2, **args)

    def max_pooling3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:4] if is_channels_last else input_shape[2:5]
        args = dict(
            pool_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else None,
            padding=self.__random.choice(["valid", "same"]),
            data_format="channels_last" if is_channels_last else "channels_first",
        )
        return 'max_pooling3D', args, self.__output_shape.pooling_layer(input_shape=input_shape, dim_num=3, **args)

    def average_pooling1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            pool_size=self.__random.kernel_size(input_shape[1:2])[0],
            strides=self.__random.sizes_with_limitation(input_shape[1:2])[0] if self.__random.boolean() else None,
            padding=self.__random.choice(["valid", "same"]),
        )
        return 'average_pooling1D', args, self.__output_shape.pooling1D_layer(input_shape=input_shape, **args)

    def average_pooling2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:3] if is_channels_last else input_shape[2:4]
        args = dict(
            pool_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else None,
            padding=self.__random.choice(["valid", "same"]),
            data_format="channels_last" if is_channels_last else "channels_first",
        )
        return 'average_pooling2D', args, self.__output_shape.pooling_layer(input_shape=input_shape, dim_num=2, **args)

    def average_pooling3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:4] if is_channels_last else input_shape[2:5]
        args = dict(
            pool_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation) if self.__random.boolean() else None,
            padding=self.__random.choice(["valid", "same"]),
            data_format="channels_last" if is_channels_last else "channels_first",
        )
        return 'average_pooling3D', args, self.__output_shape.pooling_layer(input_shape=input_shape, dim_num=3, **args)

    def global_max_pooling1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict()
        return 'global_max_pooling1D', args, self.__output_shape.global_pooling1D_layer(input_shape=input_shape, **args)

    def global_max_pooling2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        args = dict(
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'global_max_pooling2D', args, self.__output_shape.global_pool_layer(input_shape=input_shape, dim_num=2, **args)

    def global_max_pooling3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        args = dict(
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'global_max_pooling3D', args, self.__output_shape.global_pool_layer(input_shape=input_shape, dim_num=3, **args)

    def global_average_pooling1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict()
        return 'global_average_pooling1D', args, self.__output_shape.global_pooling1D_layer(input_shape=input_shape, **args)

    def global_average_pooling2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        args = dict(
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'global_average_pooling2D', args, self.__output_shape.global_pool_layer(input_shape=input_shape, dim_num=2, **args)

    def global_average_pooling3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        args = dict(
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'global_average_pooling3D', args, self.__output_shape.global_pool_layer(input_shape=input_shape, dim_num=3, **args)

    def LSTM_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            units=self.__random.ele_size(),
            activation=self.__random.activation_func(),
            recurrent_activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            recurrent_initializer='random_uniform',
            bias_initializer='random_uniform',
            unit_forget_bias=self.__random.boolean(),
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            implementation=self.__random.choice([1, 2]),
            return_sequences=self.__random.boolean(),
            return_state=False,
            go_backwards=self.__random.boolean(),
            stateful=False,  # 如果为True, batch_size需要给出
            unroll=self.__random.boolean(),
        )
        return 'LSTM', args, self.__output_shape.RNN(input_shape=input_shape, **args)

    def GRU_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            units=self.__random.ele_size(),
            activation=self.__random.activation_func(),
            recurrent_activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            recurrent_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            implementation=self.__random.choice([1, 2]),
            return_sequences=self.__random.boolean(),
            return_state=False,
            go_backwards=self.__random.boolean(),
            stateful=False,
            unroll=self.__random.boolean(),
            reset_after=self.__random.boolean(),
        )
        return 'GRU', args, self.__output_shape.RNN(input_shape=input_shape, **args)

    def simpleRNN_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            units=self.__random.ele_size(),
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            recurrent_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=self.__random.boolean(),
            return_state=False,
            go_backwards=self.__random.boolean(),
            stateful=False,
            unroll=self.__random.boolean(),
        )
        return 'simpleRNN', args, self.__output_shape.RNN(input_shape=input_shape, **args)

    def time_distributed_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3+D向量
        '''
        inner_layer_type, inner_layer_args, inner_layer_output_shape = self.__generator.generate((input_shape[0], *input_shape[2:],))
        args = dict(
            layer=dict(type=inner_layer_type, args=inner_layer_args)  # 由json生成模型时注意递归生成
        )
        return 'time_distributed', args, self.__output_shape.time_distributed_layer(input_shape=input_shape, inner_layer_output_shape=inner_layer_output_shape)

    def bidirectional_layer(self, input_shape: Tuple[Optional[int]]):
        '''包装RNN层
        '''
        inner_layer_type, inner_layer_args, inner_layer_output_shape = self.__generator.generate_RNN_layer(input_shape)
        args = dict(
            layer=dict(type=inner_layer_type, args=inner_layer_args),
            merge_mode=self.__random.choice(['sum', 'mul', 'concat', 'ave']),  # None的话会返回list，链式模型中不适合
            weights=None,  # 未知用途的参数
        )
        return 'bidirectional', args, self.__output_shape.bidirectional_layer(inner_layer_output_shape=inner_layer_output_shape, **args)

    def convLSTM2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        kernel_size, strides, data_format, dilation_rate = self.__random.conv_args((*input_shape[:1], *input_shape[2:]), dim_num=2)
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=kernel_size,
            strides=strides,
            padding=self.__random.choice(["valid", "same"]),
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=self.__random.activation_func(),
            recurrent_activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            recurrent_initializer='random_uniform',
            bias_initializer='random_uniform',
            unit_forget_bias=self.__random.boolean(),
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            return_sequences=self.__random.boolean(),
            go_backwards=False,
            stateful=False,
            dropout=0.0,
            recurrent_dropout=0.0,
        )
        return 'convLSTM2D', args, self.__output_shape.convLSTM2D_layer(input_shape=input_shape, **args)

    def batch_normalization_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            axis=self.__random.axis(len(input_shape)),
            momentum=self.__random.small_val(),
            epsilon=self.__random.small_val(),
            center=self.__random.boolean(),
            scale=self.__random.boolean(),
            beta_initializer='random_uniform',
            gamma_initializer='random_uniform',
            moving_mean_initializer='random_uniform',
            moving_variance_initializer='random_uniform',
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
        )
        return 'batch_normalization', args, self.__output_shape.batch_normalization_layer(input_shape=input_shape)

    def reshape_layer(self, input_shape: Tuple[Optional[int]], output_shape: Optional[Tuple[Optional[int]]] = None):
        if output_shape is None:
            output_shape = self.__random.target_shape(input_shape[1:])
        else:
            output_shape = output_shape[1:]
        args = dict(
            target_shape=output_shape
        )
        return 'reshape', args, self.__output_shape.reshape_layer(**args)

    def flatten_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3+D向量
        '''
        args = dict(
            data_format=self.__random.choice(["channels_last", "channels_first"]) if self.__random.boolean() else None,
        )
        return 'flatten', args, self.__output_shape.flatten_layer(input_shape=input_shape)

    def repeat_vector_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入2D向量
        '''
        args = dict(
            n=self.__random.ele_size()
        )
        return 'repeat_vector', args, self.__output_shape.repeat_vector_layer(input_shape=input_shape, **args)

    def permute_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            dims=self.__random.permute_dim_list(len(input_shape)),
        )
        return 'permute', args, self.__output_shape.permute_layer(input_shape=input_shape, **args)

    def cropping1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        a = self.__random.randint_in_range([0, input_shape[1]-1])
        b = self.__random.randint_in_range([0, input_shape[1]-1-a])
        args = dict(
            cropping=self.__random.choice([min(a, b), (a, b)]),
        )
        return 'cropping1D', args, self.__output_shape.cropping_layer(input_shape=input_shape, dim_num=1, data_format="channels_last", **args)

    def cropping2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        channels_last = self.__random.boolean()
        idx = 1 if channels_last else 2
        a = self.__random.randint_in_range([0, input_shape[idx]-1])
        b = self.__random.randint_in_range([0, input_shape[idx]-1-a])
        c = self.__random.randint_in_range([0, input_shape[idx+1]-1])
        d = self.__random.randint_in_range([0, input_shape[idx+1]-1-c])
        args = dict(
            cropping=self.__random.choice([min(min(a, b), min(c, d)),
                                          (min(a, b), min(c, d)),
                                          ((a, b), (c, d))]),
            data_format='channels_last' if channels_last else 'channels_first',
        )
        return 'cropping2D', args, self.__output_shape.cropping_layer(input_shape=input_shape, dim_num=2, **args)

    def cropping3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        channels_last = self.__random.boolean()
        idx = 1 if channels_last else 2
        a = self.__random.randint_in_range([0, input_shape[idx]-1])
        b = self.__random.randint_in_range([0, input_shape[idx]-1-a])
        c = self.__random.randint_in_range([0, input_shape[idx+1]-1])
        d = self.__random.randint_in_range([0, input_shape[idx+1]-1-c])
        e = self.__random.randint_in_range([0, input_shape[idx+2]-1])
        f = self.__random.randint_in_range([0, input_shape[idx+2]-1-e])
        args = dict(
            cropping=self.__random.choice([min(min(a, b), min(c, d), min(e, f)),
                                           (min(a, b), min(c, d), min(e, f)),
                                           ((a, b), (c, d), (e, f))]),
            data_format='channels_last' if channels_last else 'channels_first',
        )
        return 'cropping3D', args, self.__output_shape.cropping_layer(input_shape=input_shape, dim_num=3, **args)

    def up_sampling1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        args = dict(
            size=self.__random.ele_size()
        )
        return 'up_sampling1D', args, self.__output_shape.up_sampling_layer(input_shape=input_shape, dim_num=1, data_format="channels_last", **args)

    def up_sampling2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        a = self.__random.ele_size()
        b = self.__random.ele_size()
        args = dict(
            size=self.__random.choice([a, (a, b)]),
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'up_sampling2D', args, self.__output_shape.up_sampling_layer(input_shape=input_shape, dim_num=2, **args)

    def up_sampling3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        a = self.__random.ele_size()
        b = self.__random.ele_size()
        c = self.__random.ele_size()
        args = dict(
            size=self.__random.choice([a, (a, b, c)]),
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'up_sampling3D', args, self.__output_shape.up_sampling_layer(input_shape=input_shape, dim_num=3, **args)

    def zero_padding1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        a = self.__random.ele_size()
        b = self.__random.ele_size()
        args = dict(
            padding=self.__random.choice([a, (a, b)]),
        )
        return 'zero_padding1D', args, self.__output_shape.zero_padding_layer(input_shape=input_shape, dim_num=1, data_format="channels_last", **args)

    def zero_padding2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        a = self.__random.ele_size()
        b = self.__random.ele_size()
        c = self.__random.ele_size()
        d = self.__random.ele_size()
        args = dict(
            padding=self.__random.choice([a, (a, c), ((a, b), (c, d))]),
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'zero_padding2D', args, self.__output_shape.zero_padding_layer(input_shape=input_shape, dim_num=2, **args)

    def zero_padding3D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入5D向量
        '''
        a = self.__random.ele_size()
        b = self.__random.ele_size()
        c = self.__random.ele_size()
        d = self.__random.ele_size()
        e = self.__random.ele_size()
        f = self.__random.ele_size()
        args = dict(
            padding=self.__random.choice([a, (a, c, e), ((a, b), (c, d), (e, f))]),
            data_format=self.__random.choice(["channels_last", "channels_first"]),
        )
        return 'zero_padding3D', args, self.__output_shape.zero_padding_layer(input_shape=input_shape, dim_num=3, **args)

    def locally_connected1D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入3D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:2]  # if is_channels_last else input_shape[2:3]
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation),  # 貌似不支持dilation_rate
            padding='valid',  # 不支持其它
            data_format="channels_last" if is_channels_last else "channels_first",
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'locally_connected1D', args, self.__output_shape.locally_connected_layer(input_shape=input_shape, dim_num=1, **args)

    def locally_connected2D_layer(self, input_shape: Tuple[Optional[int]]):
        '''只允许输入4D向量
        '''
        is_channels_last = self.__random.boolean()
        window_limitation = input_shape[1:3] if is_channels_last else input_shape[2:4]
        args = dict(
            filters=self.__random.ele_size(),
            kernel_size=self.__random.kernel_size(window_limitation),
            strides=self.__random.sizes_with_limitation(window_limitation),  # 貌似不支持dilation_rate
            padding='valid',  # 不支持其它
            data_format="channels_last" if is_channels_last else "channels_first",
            activation=self.__random.activation_func(),
            use_bias=self.__random.boolean(),
            kernel_initializer='random_uniform',
            bias_initializer='random_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
        )
        return 'locally_connected2D', args, self.__output_shape.locally_connected_layer(input_shape=input_shape, dim_num=2, **args)

    def ReLU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            max_value=self.__random.val_size(must_positive=True) if self.__random.boolean() else None,
        )
        return 'ReLU', args, self.__output_shape.activation_layer(input_shape)

    def softmax_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            axis=self.__random.axis(len(input_shape)),
        )
        return 'softmax', args, self.__output_shape.activation_layer(input_shape)

    def leakyReLU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            alpha=self.__random.small_val(),
        )
        return 'leakyReLU', args, self.__output_shape.activation_layer(input_shape)

    def PReLU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            alpha_initializer='random_uniform',
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=self.__random.axis_list(len(input_shape)) if self.__random.boolean() else None,
        )
        return 'PReLU', args, self.__output_shape.activation_layer(input_shape)

    def ELU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            alpha=self.__random.small_val(),
        )
        return 'ELU', args, self.__output_shape.activation_layer(input_shape)

    def thresholded_ReLU_layer(self, input_shape: Tuple[Optional[int]]):
        args = dict(
            theta=self.__random.small_val(),
        )
        return 'thresholded_ReLU', args, self.__output_shape.activation_layer(input_shape)

    def concatenate_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, axis = self.__random.concatenate_shapes(input_num=input_num, output_shape=output_shape)
        args = dict(
            axis=axis
        )
        return 'concatenate', args, input_shape_list, output_shape

    def average_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, _ = self.__random.normal_merge_shapes(input_num=input_num, output_shape=output_shape)
        return 'average', dict(), input_shape_list, output_shape

    def maximum_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, _ = self.__random.normal_merge_shapes(input_num=input_num, output_shape=output_shape)
        return 'maximum', dict(), input_shape_list, output_shape

    def minimum_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, _ = self.__random.normal_merge_shapes(input_num=input_num, output_shape=output_shape)
        return 'minimum', dict(), input_shape_list, output_shape

    def add_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, _ = self.__random.normal_merge_shapes(input_num=input_num, output_shape=output_shape)
        return 'add', dict(), input_shape_list, output_shape

    def subtract_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, _ = self.__random.normal_merge_shapes(input_num=input_num, output_shape=output_shape)
        return 'subtract', dict(), input_shape_list, output_shape

    def multiply_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, _ = self.__random.normal_merge_shapes(input_num=input_num, output_shape=output_shape)
        return 'multiply', dict(), input_shape_list, output_shape

    def dot_layer(self, input_num: int, output_shape):
        input_shape_list, output_shape, axes = self.__random.dot_shapes()
        args = dict(
            axes=axes,
            normalize=self.__random.boolean(),
        )
        return 'dot', args, input_shape_list, output_shape


if __name__ == '__main__':
    pass
