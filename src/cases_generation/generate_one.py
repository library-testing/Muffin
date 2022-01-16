import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings

from utils.utils import switch_backend, get_layer_func

warnings.filterwarnings("ignore")


def __generate_layer(layer_info: dict, weight_range: Tuple[float]):
    # 解析层信息
    layer_type, layer_args, pre_layers, ouput_shape = tuple(map(layer_info.get, ['type', 'args', 'pre_layers', 'output_shape']))
    layer = get_layer_func(layer_type)

    # 解析参数
    for k, v in layer_args.items():
        if k == 'layer':
            layer_args[k], _, _ = __generate_layer(v, weight_range)  # 嵌套层需要递归生成
        elif k[-12:] == '_initializer':
            a, b = weight_range
            if k == 'moving_variance_initializer':
                layer_args[k] = keras.initializers.random_uniform(0, b)
            else:
                layer_args[k] = keras.initializers.random_uniform(a, b)  # initializer

    return layer(**layer_args), pre_layers, ouput_shape


def __generate_model(json_path: str, weight_range: Tuple[float]):
    # 加载模型结构数据
    with open(json_path, 'r') as f:
        model_info = json.load(f)

    input_id_list, output_id_list = model_info['input_id_list'], model_info['output_id_list']

    input_list, output_list, layer_dict = [], [], {}
    for layer_id, layer_info in model_info['model_structure'].items():  # 按拓扑排序遍历
        layer_id = int(layer_id)
        # 生成层
        layer, inbound_layers_idx, ouput_shape = __generate_layer(layer_info, weight_range)

        # 层拼接
        if layer_id in input_id_list:
            layer_dict[layer_id] = layer  # input_object
            input_list.append(layer_dict[layer_id])

        else:
            inbound_layers = [layer_dict[i] for i in inbound_layers_idx]
            layer_dict[layer_id] = layer(inbound_layers[0] if len(inbound_layers) == 1 else inbound_layers)  # 对layers进行连接

        if layer_id in output_id_list:
            output_list.append(layer_dict[layer_id])

        # 检查形状
        from keras import backend as K
        if K.int_shape(layer_dict[layer_id]) != tuple(ouput_shape):
            raise Exception(f"[Debug] layer_id: {layer_id} expected shape: {tuple(ouput_shape)}  actual shape: {K.int_shape(layer_dict[layer_id])}")

    return keras.Model(inputs=input_list, outputs=output_list)


def __save_weights(model, weights_dir, bk):
    if bk == 'theano':
        keras.utils.convert_all_kernels_in_model(model)
    for layer in model.layers:
        save_path = Path(weights_dir) / f'{layer.name}.npz'
        np.savez(save_path, *layer.get_weights())


if __name__ == "__main__":
    # 获取参数
    parse = argparse.ArgumentParser()
    parse.add_argument("--backend", type=str)
    parse.add_argument("--json_path", type=str)
    parse.add_argument("--weight_minv", type=float)
    parse.add_argument("--weight_maxv", type=float)
    parse.add_argument("--output_dir", type=str)
    flags, _ = parse.parse_known_args(sys.argv[1:])

    try:
        switch_backend(flags.backend)  # 切换后端
        import keras

        model = __generate_model(flags.json_path, (flags.weight_minv, flags.weight_maxv))

        # 保存模型
        model_path = Path(flags.output_dir) / f'{flags.backend}.h5'
        model.save(str(model_path), include_optimizer=False)

        # # 保存图片
        # pic_path = Path(flags.output_dir) / f'{flags.backend}.png'
        # keras.utils.plot_model(model, str(pic_path), show_shapes=True)

        # 保存模型权重
        weights_dir = Path(flags.output_dir) / 'initial_weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        __save_weights(model, weights_dir, flags.backend)

    except Exception:
        import traceback

        # 创建log文件
        log_dir = Path(flags.output_dir).parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        with (log_dir / 'generation.log').open(mode='a', encoding='utf-8') as f:
            f.write(f"[ERROR] Fail when generating model with {flags.backend}\n")
            traceback.print_exc(file=f)
            f.write("\n\n")

        sys.exit(-1)
