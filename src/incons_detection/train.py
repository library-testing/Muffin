import argparse
import sys
import json
from pathlib import Path
import numpy as np
import warnings

from utils.utils import switch_backend

warnings.filterwarnings("ignore")


def __prepare(loss_type: str, optimizer_type: str, training_instances_path: str, ground_truths_path: str, model_path: str, model_info_path: str):
    # 加载模型和数据
    model = keras.models.load_model(model_path)
    with open(model_info_path, "r") as f:
        model_info = json.load(f)
        input_objects_names = [model_info['model_structure'][str(idx)]['args']['name'] for idx in model_info["input_id_list"]]
        output_layers_names = [model_info['model_structure'][str(idx)]['args']['name'] for idx in model_info["output_id_list"]]
    tmp = np.load(training_instances_path)
    training_instances = [*tmp.values()]
    ground_truths = [*np.load(ground_truths_path).values()]
    model.compile(loss=loss_type, optimizer=optimizer_type)  # 使用相同的训练配置

    # feed数据
    ins = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    x, y, sample_weight = model._standardize_user_data(training_instances, ground_truths)
    ins_value = x + y + sample_weight
    return model, input_objects_names, output_layers_names, training_instances, ground_truths, ins, ins_value


def __get_outputs(model, input_objects_names, x,
                  output_dir: str):
    # 获取所有层的输出
    get_layer_output = K.function(model._feed_inputs + [K.learning_phase()],
                                  [layer.output for layer in model.layers if layer.name not in input_objects_names])
    
    layers_names = [layer.name for layer in model.layers if layer.name not in input_objects_names]
    layers_outputs = get_layer_output(x + [1])

    # 保存各层outputs
    def save_outputs(layers_names, layers_outputs, output_dir):
        for name, output in zip(layers_names, layers_outputs):
            save_path = Path(output_dir) / f'{name}.npy'
            np.save(save_path, output)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_outputs(input_objects_names + layers_names, x + layers_outputs, output_dir)

    return {layer_name: output for layer_name, output in zip(input_objects_names + layers_names, x + layers_outputs)}


def __get_loss(model, ins, ins_value,
               loss_path: str):
    # 获取loss的function
    get_loss = K.function(
        ins + [K.learning_phase()],
        [model.total_loss]
    )
    loss_value = get_loss(ins_value + [1])[0]

    # 保存loss
    def save_loss(loss, output_path):
        with open(output_path, 'w') as f:
            f.write(str(loss))
    save_loss(loss_value, loss_path)


def __get_loss_gradients(model, output_layers_names, ins, ins_value, layers_outputs_value, y,
                         loss_grads_dir: str, model_info_path: str):
    layer_outputs = [model.get_layer(layer_name).output for layer_name in output_layers_names]

    # 获取d(loss)/d(output)
    if K.backend() == 'cntk':
        loss_grads_value = [__cntk_get_gradients(model, layer_name, layers_outputs_value, y, model_info_path) for layer_name in output_layers_names]

    else:
        get_loss_grads = K.function(
            ins + [K.learning_phase()],
            K.gradients(model.total_loss, layer_outputs)
        )
        loss_grads_value = get_loss_grads(ins_value + [1])

    # 保存d(loss)/d(output)
    def save_loss_grad(layer_names, grads_value, output_dir):
        for layer_name, g in zip(layer_names, grads_value):
            save_path = Path(output_dir) / f'{layer_name}.npy'
            np.save(save_path, g)

    Path(loss_grads_dir).mkdir(parents=True, exist_ok=True)
    save_loss_grad(output_layers_names, loss_grads_value, loss_grads_dir)


def __get_gradients(model, input_objects_names, ins, ins_value, layers_outputs_value, y,
                    grads_dir: str, model_info_path: str):
    layer_names = input_objects_names + [layer.name for layer in model.layers if layer.name not in input_objects_names]

    # 获取gradients的function
    if K.backend() == 'cntk':
        grads = [__cntk_get_gradients(model, layer_name, layers_outputs_value, y, model_info_path) for layer_name in layer_names]

    else:
        layer_outputs = model.inputs + [layer.output for layer in model.layers if layer.name not in input_objects_names]
        get_gradients = K.function(
            ins + [K.learning_phase()],
            K.gradients(model.total_loss, layer_outputs)
        )
        grads = get_gradients(ins_value + [1])

    # 保存训练中的gradients值
    def save_gradients(layer_names, grads, output_dir):
        for name, g in zip(layer_names, grads):
            save_path = Path(output_dir) / f'{name}.npy'
            np.save(save_path, g)

    Path(grads_dir).mkdir(parents=True, exist_ok=True)
    save_gradients(layer_names, grads, grads_dir)


# def __get_weights(model, x, y,
#                   output_dir: str):

#     # 训练
#     model.train_on_batch(x=x, y=y)

#     # 保存训练后的weights
#     def save_weights(model, output_dir):
#         for p in model.trainable_weights:
#             save_path = Path(output_dir) / f'{p.name.replace("/", "-").replace(":", "-")}.npy'
#             np.save(save_path, K.get_value(p))

#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     save_weights(model, output_dir)


def __cntk_get_gradients(model, layer_name, layer_outputs_value, y_true_list, model_info_path):
    with open(str(model_info_path), "r") as f:
        model_info = json.load(f)

    import cntk as C
    _output = C.input_variable(model.get_layer(layer_name).output_shape[1:], needs_gradient=True)
    tmp_input = keras.Input(tensor=_output)

    tmp_inputs = []
    extra_inputs = []
    extra_input_datas = []
    layer_outputs = {}

    def get_inbound_layers(layer):
        ids = model_info['model_structure'][str(int(layer.name[:2]))]['pre_layers']
        names = [model_info['model_structure'][str(idx)]['args']['name'] for idx in ids]
        return [model.get_layer(name) for name in names]

    def get_output_of_layer(layer):
        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        if layer.name == layer_name:
            tmp_inputs.append(tmp_input)
            layer_outputs[layer.name] = tmp_input
            return tmp_input

        # 提取其余input层
        if int(layer.name[:2]) < int(layer_name[:2]):
            _input = C.input_variable(layer.output_shape[1:], needs_gradient=False)
            tmp = keras.Input(tensor=_input)

            extra_inputs.append(_input)
            extra_input_datas.append(layer_outputs_value[layer.name])  # 取其它链的输入数据

            tmp_inputs.append(tmp)
            layer_outputs[layer.name] = tmp
            return tmp

        inbound_layers = get_inbound_layers(layer)
        layer_ins = [get_output_of_layer(layer) for layer in inbound_layers]

        out = layer(layer_ins[0] if len(layer_ins) == 1 else layer_ins)
        layer_outputs[layer.name] = out
        return out

    # 提取output层
    tmp_outputs = []
    for output_id in model_info['output_id_list']:
        name = model_info['model_structure'][str(output_id)]['args']['name']
        tmp_outputs.append(get_output_of_layer(model.get_layer(name)))
    tmp_model = keras.models.Model(inputs=tmp_inputs, outputs=tmp_outputs)

    x = [layer_outputs_value[layer_name], *extra_input_datas]
    y_true = y_true_list

    tmp_model.compile(loss=model.loss, optimizer=model.optimizer)

    ins = [_output, *extra_inputs] + tmp_model._feed_targets + tmp_model._feed_sample_weights
    _, y, sample_weight = tmp_model._standardize_user_data([], y_true)
    ins_value = x + y + sample_weight

    grads = tmp_model.total_loss.grad({
        k: v
        for k, v in zip(ins, ins_value)
    }, [_output])

    return grads


if __name__ == "__main__":
    # 获取参数
    parse = argparse.ArgumentParser()
    parse.add_argument("--backend", type=str)
    parse.add_argument("--loss", type=str)
    parse.add_argument("--optimizer", type=str)
    parse.add_argument("--model_path", type=str)
    parse.add_argument("--model_info_path", type=str)
    parse.add_argument("--training_instances_path", type=str)
    parse.add_argument("--ground_truths_path", type=str)
    parse.add_argument("--outputs_dir", type=str)
    parse.add_argument("--loss_path", type=str)
    parse.add_argument("--loss_grads_dir", type=str)
    parse.add_argument("--gradients_dir", type=str)
    flags, _ = parse.parse_known_args(sys.argv[1:])

    try:
        switch_backend(flags.backend)  # 切换后端
        import keras
        from keras import backend as K

        FLAG = -1
        model, input_objects_names, output_layers_names, x, y, ins, ins_value = __prepare(flags.loss, flags.optimizer, flags.training_instances_path, flags.ground_truths_path, flags.model_path, flags.model_info_path)
        FLAG = 1
        layers_outputs_value = __get_outputs(model, input_objects_names, x, flags.outputs_dir)
        FLAG = 2
        __get_loss(model, ins, ins_value, flags.loss_path)
        FLAG = 3
        __get_loss_gradients(model, output_layers_names, ins, ins_value, layers_outputs_value, y, flags.loss_grads_dir, flags.model_info_path)
        FLAG = 4
        __get_gradients(model, input_objects_names, ins, ins_value, layers_outputs_value, y, flags.gradients_dir, flags.model_info_path)
        FLAG = 5
        # __get_weights(model, x, y, flags.weights_dir)
        # FLAG = 6

        if K.backend() in ['tensorflow', 'cntk']:
            K.clear_session()

    except Exception:
        import traceback

        # 创建log文件
        log_dir = Path(flags.outputs_dir).parent.parent / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        with (log_dir / 'detection.log').open(mode='a', encoding='utf-8') as f:
            f.write(f"[ERROR] Crash when training model with {flags.backend}\n")
            traceback.print_exc(file=f)
            f.write("\n\n")

        sys.exit(FLAG)
