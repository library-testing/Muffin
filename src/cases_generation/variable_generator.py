import random
from typing import List, Tuple, Iterable, Optional


class VariableGenerator(object):
    '''随机变量生成器
    '''

    def __init__(self, config: dict):
        super().__init__()
        self.__tensor_dimen_range = config['tensor_dimension_range']
        self.__tensor_ele_size_range = config['tensor_element_size_range']
        self.__weight_val_range = config['weight_value_range']
        self.__small_val_range = config['small_value_range']
        self.__vocabulary_size = config['vocabulary_size']

    def input_object(self, shape: Optional[Tuple[Optional[int]]] = None):
        '''返回一个随机生成的input张量的名字, 参数字典和形状, 形状也可以指定
        '''
        if shape is None:
            shape = self.shape()
        args = dict(
            shape=shape[1:],
            batch_shape=None,
            dtype=None,
            sparse=False,  # 输入为稀疏矩阵时才会设置
            tensor=None,
        )
        return 'input_object', args, shape

    def choice(self, seq: list):
        return random.choice(seq)

    def choice_by_scores(self, seq: list, scores: list):
        return random.choices(seq, scores, k=1)[0]

    def shape(self, dim: Optional[int] = None) -> Tuple[int]:
        '''随机返回一个可以表示shape的元组
        '''
        if dim is None:
            dim = random.randint(*self.__tensor_dimen_range)
        return (
            None,
            *tuple(random.choices(range(self.__tensor_ele_size_range[0], self.__tensor_ele_size_range[1]+1),
                                  k=dim-1))
        )

    def target_shape(self, old_shape: Tuple[int]) -> Tuple[int]:
        '''用old_shape的shuffle来代替, 不包括batch axis
        '''
        res_shape = list(old_shape)
        random.shuffle(res_shape)
        return tuple(res_shape)

    def axis(self, input_dim: int):
        '''随即返回一个维度的idx
        '''
        return random.randint(1, input_dim-1) if self.boolean() else -1

    def axis_list(self, input_dim):
        return random.choices(range(1, input_dim), k=random.randint(1, input_dim-1))

    def permute_dim_list(self, input_dim):
        res_list = list(range(1, input_dim))
        random.shuffle(res_list)
        return res_list

    def boolean(self) -> bool:
        '''随机返回一个布尔值
        '''
        return random.random() < 0.5

    def randint_in_range(self, ran: Iterable) -> int:
        '''随机返回[a, b]中一个整数
        '''
        return random.randint(*ran)

    def ele_size(self) -> int:
        '''随机返回tensor形状元组的一个元素的数值
        '''
        return random.randint(*self.__tensor_ele_size_range)

    def dimen_size(self) -> int:
        '''随机返回一个维数
        '''
        return random.randint(*self.__tensor_dimen_range)

    def val_size(self, must_positive: bool = False) -> float:
        '''随机返回一个weight值
        '''
        a, b = self.__weight_val_range
        if must_positive:
            a = max(a, 0)
        return random.random() * (b - a) + a

    def small_val(self):
        a, b = self.__small_val_range
        return random.random() * (b - a) + a

    def vocabulary_size(self) -> int:
        '''返回vocabulary大小(Embedding层)
        '''
        return self.__vocabulary_size

    def kernel_size(self, window_max_shape: Tuple[int]) -> List[int]:
        length = random.randint(1, min(window_max_shape))
        return [length for _ in window_max_shape]

    def sizes_with_limitation(self, window_max_shape: Tuple[int]) -> List[int]:
        length = random.randint(1, min(window_max_shape))
        return [length for _ in window_max_shape]

    def activation_func(self):
        '''随机返回一个激活函数
        '''
        return random.choice([
            'relu',
            'sigmoid',
            'softmax',
            'softplus',
            # 'softsign'  # x / (abs(x) + 1) 会导致值都很接近0.99,
            'tanh',
            'selu',
            'elu',
            'linear',
        ])

    def conv_args(self, input_shape: Tuple[int], dim_num: int, is_channels_last: Optional[bool] = None):
        is_channels_last = self.boolean() if is_channels_last is None else is_channels_last
        data_format = "channels_last" if is_channels_last else "channels_first"
        window_limitation = input_shape[1:1+dim_num] if is_channels_last else input_shape[2:2+dim_num]
        kernel_size = self.kernel_size(window_limitation)
        dilation_limitation = [window_limitation[i]//kernel_size[i] for i in range(dim_num)]
        if self.boolean():
            dilation_rate = self.sizes_with_limitation(dilation_limitation)
            strides = [1 for _ in range(dim_num)]
        else:
            strides = self.sizes_with_limitation(dilation_limitation)
            dilation_rate = [1 for _ in range(dim_num)]
        return kernel_size, strides, data_format, dilation_rate

    def concatenate_shapes(self, input_num: int, output_shape: Optional[tuple]):
        if output_shape is None:
            input_shape_1 = self.shape()
            axis = self.axis(len(input_shape_1))
            temp_axis = axis % len(input_shape_1)

            input_shape_list = [input_shape_1]
            total_len = input_shape_1[temp_axis]
            for _ in range(input_num-1):
                new_len = self.ele_size()
                input_shape_list.append(
                    (*input_shape_1[:temp_axis], new_len, *input_shape_1[temp_axis+1:])
                )
                total_len += new_len
            output_shape = (*input_shape_1[:temp_axis], total_len, *input_shape_1[temp_axis+1:])
        else:
            import numpy as np
            axis = np.argmax(list(output_shape[1:])) + 1
            total_len = output_shape[axis]
            new_len_list = self.divide_len(total_len, input_num)
            input_shape_list = [(*output_shape[:axis], new_len, *output_shape[axis+1:]) for new_len in new_len_list]
        return input_shape_list, output_shape, axis

    def divide_len(self, num, _len):
        if _len < num:
            raise ValueError("len not enough.")
        res_list = [self.randint_in_range([1, _len // num]) for _ in range(num-1)]
        remain = _len - sum(res_list)
        res_list.append(remain)
        return res_list

    def normal_merge_shapes(self, input_num: int, output_shape: Optional[tuple]):
        input_shape = self.shape() if output_shape is None else output_shape
        return [input_shape for _ in range(input_num)], input_shape, None

    def dot_shapes(self):
        input_shape_1 = self.shape()
        input_shape_2 = self.shape()
        axes = (self.axis(len(input_shape_1)), self.axis(len(input_shape_2))) if self.boolean() else self.axis(min(len(input_shape_1), len(input_shape_2)))
        temp_axes = (axes, axes) if isinstance(axes, int) else axes
        temp_axes = (temp_axes[0] % len(input_shape_1), temp_axes[1] % len(input_shape_2))
        new_len = self.ele_size()
        input_shape_1 = (*input_shape_1[:temp_axes[0]], new_len, *input_shape_1[temp_axes[0]+1:])
        input_shape_2 = (*input_shape_2[:temp_axes[1]], new_len, *input_shape_2[temp_axes[1]+1:])
        output_shape = (None,
                        *input_shape_1[1:temp_axes[0]], *input_shape_1[temp_axes[0]+1:],
                        *input_shape_2[1:temp_axes[1]], *input_shape_2[temp_axes[1]+1:])
        if len(output_shape) == 1:
            output_shape = (None, 1)
        return [input_shape_1, input_shape_2], output_shape, axes


if __name__ == '__main__':
    config = {
        'tensor_dimension_range': (1, 5),
        'tensor_element_size_range': (1, 64),
        'weight_value_range': (-100000, 100000),
        'vocabulary_size': 2000,
        'kernel_size_range': (1, 3),
    }
