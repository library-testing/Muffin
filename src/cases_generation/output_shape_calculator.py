from functools import reduce


class OutputShapeCalculator(object):

    def __init__(self):
        super().__init__()

    def dense_layer(self, input_shape, units, **kwargs):
        return (*input_shape[:-1], units)

    def activation_layer(self, input_shape):
        return input_shape

    def embedding_layer(self, input_shape, output_dim, **kwargs):
        return (*input_shape, output_dim)

    def masking_layer(self, input_shape):
        return input_shape

    def conv_layer(self, input_shape, dim_num, data_format, kernel_size, dilation_rate, padding, strides, filters, **kwargs):
        is_channels_last = (data_format == "channels_last")
        old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]
        window_len = [(kernel_size[i] - 1) * dilation_rate[i] + 1 for i in range(dim_num)]
        plus = [0 if padding == 'valid' else window_len[i] - 1 for i in range(dim_num)]
        new_steps = [(old_steps[i] + plus[i] - window_len[i]) // strides[i] + 1 for i in range(dim_num)]
        return (*input_shape[:-1-dim_num], *new_steps, filters) if is_channels_last else (*input_shape[:-dim_num-1], filters, *new_steps)

    def depthwise_conv2D_layer(self, input_shape, data_format, depth_multiplier, **kwargs):
        filters_in = input_shape[-1] if data_format == "channels_last" else input_shape[-3]
        return self.conv_layer(input_shape, 2, data_format, filters=filters_in * depth_multiplier, dilation_rate=[1, 1], **kwargs)

    def deconv_length(self, dim_size, stride_size, kernel_size, padding, dilation=1):
        kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        if padding == 'valid':
            dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
        elif padding == 'same':
            dim_size = dim_size * stride_size
        return dim_size

    def conv_transpose_layer(self, input_shape, dim_num, data_format, strides, kernel_size, padding, filters, **kwargs):
        is_channels_last = (data_format == "channels_last")
        old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]
        new_steps = [self.deconv_length(old_steps[i], strides[i], kernel_size[i], padding) for i in range(dim_num)]
        return (*input_shape[:-1-dim_num], *new_steps, filters) if is_channels_last else (*input_shape[:-dim_num-1], filters, *new_steps)

    def pooling_layer(self, input_shape, dim_num, pool_size, strides, padding, data_format, **kwargs):
        is_channels_last = (data_format == "channels_last")
        old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]
        plus = [0 if padding == 'valid' else pool_size[i] - 1 for i in range(dim_num)]
        strides = pool_size if strides is None else strides
        new_steps = [(old_steps[i] + plus[i] - pool_size[i]) // strides[i] + 1 for i in range(dim_num)]
        return (*input_shape[:-1-dim_num], *new_steps, input_shape[-1]) if is_channels_last else (*input_shape[:-dim_num], *new_steps)

    def pooling1D_layer(self, input_shape, pool_size, strides, padding, **kwargs):
        return self.pooling_layer(input_shape, 1, [pool_size], [pool_size] if strides is None else [strides], padding, "channels_last", **kwargs)

    def global_pool_layer(self, input_shape, dim_num, data_format, **kwargs):
        is_channels_last = (data_format == "channels_last")
        return (*input_shape[:-1-dim_num], input_shape[-1]) if is_channels_last else input_shape[:-dim_num]

    def global_pooling1D_layer(self, input_shape, **kwargs):
        return self.global_pool_layer(input_shape, 1, "channels_last", **kwargs)

    def RNN(self, input_shape, return_sequences, units, **kwargs):
        return (*input_shape[:-1], units) if return_sequences else (*input_shape[:-2], units)

    def time_distributed_layer(self, input_shape, inner_layer_output_shape):
        return (*input_shape[:2], *inner_layer_output_shape[1:])

    def bidirectional_layer(self, inner_layer_output_shape, merge_mode, **kwargs):
        return (*inner_layer_output_shape[:-1], 2 * inner_layer_output_shape[-1]) if merge_mode == 'concat' else inner_layer_output_shape

    def convLSTM2D_layer(self, input_shape, return_sequences, **kwargs):
        conv_output_shape = self.conv_layer((*input_shape[:1], *input_shape[2:]), 2, **kwargs)
        return (*input_shape[:2], *conv_output_shape[1:]) if return_sequences else (*input_shape[:1], *conv_output_shape[1:])

    def batch_normalization_layer(self, input_shape):
        return input_shape

    def reshape_layer(self, target_shape, **kwargs):
        return (None, *target_shape)

    def flatten_layer(self, input_shape):
        return (None, reduce(lambda x, y: x*y, input_shape[1:])) if len(input_shape) >= 2 else (None, 1)

    def repeat_vector_layer(self, input_shape, n, **kwargs):
        return (*input_shape[:-1], n, input_shape[-1])

    def permute_layer(self, input_shape, dims, **kwargs):
        return (None, *[input_shape[idx] for idx in dims])

    def cropping_layer(self, input_shape, dim_num, data_format, cropping, **kwargs):
        is_channels_last = (data_format == "channels_last")
        old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]
        if isinstance(cropping, int):
            cropping = [(cropping, cropping) for _ in range(dim_num)]
        elif isinstance(cropping[0], int):
            cropping = [(v, v) for v in cropping] if dim_num >= 2 else [cropping]
        new_steps = [old_steps[i] - sum(cropping[i]) for i in range(dim_num)]
        return (*input_shape[:-1-dim_num], *new_steps, input_shape[-1]) if is_channels_last else (*input_shape[:-dim_num], *new_steps)

    def up_sampling_layer(self, input_shape, dim_num, data_format, size, **kwargs):
        is_channels_last = (data_format == "channels_last")
        old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]
        if isinstance(size, int):
            size = [size for _ in range(dim_num)]
        new_steps = [old_steps[i] * size[i] for i in range(dim_num)]
        return (*input_shape[:-1-dim_num], *new_steps, input_shape[-1]) if is_channels_last else (*input_shape[:-dim_num], *new_steps)

    def zero_padding_layer(self, input_shape, dim_num, data_format, padding, **kwargs):
        is_channels_last = (data_format == "channels_last")
        old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]
        if isinstance(padding, int):
            padding = [(padding, padding) for _ in range(dim_num)]
        elif isinstance(padding[0], int):
            padding = [(v, v) for v in padding] if dim_num >= 2 else [padding]
        new_steps = [old_steps[i] + sum(padding[i]) for i in range(dim_num)]
        return (*input_shape[:-1-dim_num], *new_steps, input_shape[-1]) if is_channels_last else (*input_shape[:-dim_num], *new_steps)

    def locally_connected_layer(self, input_shape, dim_num, data_format, kernel_size, padding, strides, filters, **kwargs):
        is_channels_last = (dim_num == 1 or data_format == "channels_last")  # dim_num=1时貌似忽略data_format参数
        old_steps = input_shape[-1-dim_num:-1] if is_channels_last else input_shape[-dim_num:]
        plus = [0 if padding == 'valid' else kernel_size[i] - 1 for i in range(dim_num)]
        new_steps = [(old_steps[i] + plus[i] - kernel_size[i]) // strides[i] + 1 for i in range(dim_num)]
        return (*input_shape[:-1-dim_num], *new_steps, filters) if is_channels_last else (*input_shape[:-dim_num-1], filters, *new_steps)
