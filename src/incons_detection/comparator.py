from itertools import combinations
import numpy as np
from pathlib import Path
import json


class Comparator(object):

    def __init__(self, db_manager):
        super().__init__()
        self.__db_manager = db_manager
        self.__log_dir = Path(".")

    def compare(self, model_id: int, exp_dir: str, backends_outputs: dict, backends_losses: dict, backends_loss_grads: dict, backends_grads: dict, ok_backends: list):
        self.__log_dir = Path(exp_dir) / 'logs'

        # 加载模型结构信息
        json_path = Path(exp_dir) / 'models' / 'model.json'
        with open(str(json_path), 'r') as f:
            model_info = json.load(f)
        model_structure = model_info['model_structure']
        output_id_list = model_info['output_id_list']
        layer_names = [layer_info['args']['name'] for layer_info in model_structure.values()]

        for bk1, bk2 in combinations(ok_backends, 2):
            outputs1, outputs2 = backends_outputs.get(bk1, None), backends_outputs.get(bk2, None)
            loss1, loss2 = backends_losses.get(bk1, None), backends_losses.get(bk2, None)
            loss_grads1, loss_grads2 = backends_loss_grads.get(bk1, None), backends_loss_grads.get(bk2, None)
            grads1, grads2 = backends_grads.get(bk1, None), backends_grads.get(bk2, None)
            # weights1, weights2 = backends_weights.get(bk1, None), backends_weights.get(bk2, None)

            # 计算model_output_delta、loss_delta、loss_grads_delta、weights_delta
            model_output_delta = loss_delta = loss_grads_delta = weights_delta = None
            try:
                model_output_delta = self.__model_output_delta(outputs1, outputs2, output_id_list)
                loss_delta = self.__loss_delta(loss1, loss2)
                loss_grads_delta = self.__loss_grads_delta(loss_grads1, loss_grads2)
                # weights_delta = self.__weights_delta(layer_names, weights1, weights2)
            except Exception:
                import traceback
                # 创建log文件
                self.__log_dir.mkdir(parents=True, exist_ok=True)
                with (self.__log_dir / 'comparation.log').open(mode='a', encoding='utf-8') as f:
                    f.write(f"[ERROR] Crash when calculate inconsistencies between {bk1} and {bk2}\n")
                    traceback.print_exc(file=f)
                    f.write("\n\n")

            # 结果存入数据库
            incons_id = self.__db_manager.add_training_incons(model_id, f'{bk1}_{bk2}', model_output_delta, loss_delta, loss_grads_delta)

            # 对每一层计算output_delta、gradients_delta及对应的差异变化率 以及 weights_delta, 并存入数据库
            outputs_map = {}
            try:
                outputs_map = self.__layers_outputs_gradients_delta(model_structure, outputs1, outputs2, grads1, grads2, loss_grads_delta)

            except Exception:
                import traceback
                # 创建log文件
                self.__log_dir.mkdir(parents=True, exist_ok=True)
                with (self.__log_dir / 'comparation.log').open(mode='a', encoding='utf-8') as f:
                    f.write(f"[ERROR] Crash when calculate localization map between {bk1} and {bk2}\n")
                    traceback.print_exc(file=f)
                    f.write("\n\n")

            # 结果存入数据库
            self.__db_manager.add_localization_map([(incons_id, layer_name, *infos) for layer_name, infos in outputs_map.items()])

    def __max_delta(self, x, y):
        # 距离公式使用max
        if x is None or y is None:
            return None
        return float(np.max(np.abs(x-y)))

    def __model_output_delta(self, outputs1, outputs2, output_id_list):
        if outputs1 is None or outputs2 is None:
            return None
        output_deltas = []
        for layer_name, o1, o2 in zip(outputs1.keys(), outputs1.values(), outputs2.values()):
            if int(layer_name[:2]) in output_id_list:
                output_deltas.append(self.__max_delta(o1, o2))
        return max(output_deltas)

    def __loss_delta(self, loss1, loss2):
        if loss1 is None or loss2 is None:
            return None
        return np.abs(loss1 - loss2)

    def __loss_grads_delta(self, grads1, grads2):
        if grads1 is None or grads2 is None:
            return None
        return max([self.__max_delta(v1, v2) for v1, v2 in zip(grads1.values(), grads2.values())])


    # def __weights_delta(self, layer_names, weights1, weights2):
    #     if weights1 is None or weights2 is None:
    #         return None
    #     weights_deltas = []
    #     for layer_name in layer_names:
    #         w1 = [w for name, w in weights1.items() if layer_name in name]
    #         w2 = [w for name, w in weights2.items() if layer_name in name]
    #         if w1 and w2:
    #             weights_deltas.append(max([self.__max_delta(v1, v2) for v1, v2 in zip(w1, w2)]))
    #     return max(weights_deltas) if weights_deltas else 0

    def __layers_outputs_gradients_delta(self, model_structure, outputs1, outputs2, grads1, grads2, loss_grads_delta, epsilon=1e-7):
        localization_map = {}

        def seek_next_layer_name(cur_layer_idx):
            for layer_info in list(model_structure.values())[cur_layer_idx+1:]:
                if cur_layer_idx in layer_info['pre_layers']:
                    return layer_info['args']['name']
            return None

        # def params_delta(grads1, grads2, layer_name):
        #     if grads1 is None and grads2 is None:
        #         return None
        #     g1_list = [g for n, g in grads1.items() if layer_name in n]
        #     g2_list = [g for n, g in grads2.items() if layer_name in n]
        #     return max([self.__max_delta(g1, g2) for g1, g2 in zip(g1_list, g2_list)]) if g1_list and g2_list else 0

        for layer_idx, layer_info in model_structure.items():
            layer_name = layer_info['args']['name']
            inbound_layer_names = [model_structure[str(index)]['args']['name'] for index in layer_info['pre_layers']]
            next_layer_name = seek_next_layer_name(int(layer_idx))

            output_delta = output_R = grad_delta = grad_R = weights_delta = None

            if outputs1 is not None and outputs2 is not None:
                o1, o2 = outputs1[layer_name], outputs2[layer_name]
                output_delta = self.__max_delta(o1, o2)
                if not inbound_layer_names:  # 无pre层
                    output_delta_pre = 0
                else:
                    delta_pre_list = []
                    for inbound_layer_name in inbound_layer_names:
                        # 计算每一pre层hidden_state的delta
                        pre_o1, pre_o2 = outputs1[inbound_layer_name], outputs2[inbound_layer_name]
                        delta_pre_list.append(self.__max_delta(pre_o1, pre_o2))
                    output_delta_pre = max(delta_pre_list)

                output_R = (output_delta - output_delta_pre) / (output_delta_pre + epsilon)

            if grads1 is not None and grads2 is not None:
                g1, g2 = grads1[layer_name], grads2[layer_name]

                grad_delta = self.__max_delta(g1, g2)
                # weights_delta = params_delta(weights1, weights2, layer_name)


                if not next_layer_name:  # 无next层，即最后一层的R是计算相对于loss_grads的变化率
                    grad_delta_next = loss_grads_delta
                else:
                    next_g1, next_g2 = grads1[next_layer_name], grads2[next_layer_name]
                    grad_delta_next = self.__max_delta(next_g1, next_g2)

                grad_R = (grad_delta - grad_delta_next) / (grad_delta_next + epsilon)

            localization_map[layer_name] = (output_delta, output_R, grad_delta, grad_R, str(inbound_layer_names))
        return localization_map

    # def __weights_output_delta(self, model_structure, grads1, grads2, weights1, weights2, epsilon=1e-7):
    #     localization_map = {}

    #     def has_weights(layer_name):
    #         for n in grads1.keys():
    #             if layer_name in n:
    #                 return True
    #         return False

    #     def seek_forward_layer_name(cur_layer_name):

    #         def is_connected(i, j):
    #             pre_layer_idxs = model_structure[str(i)]['pre_layers']
    #             if j in pre_layer_idxs:
    #                 return True
    #             else:
    #                 for k in pre_layer_idxs:
    #                     if is_connected(k, j):
    #                         return True
    #                 return False

    #         cur_index = int(cur_layer_name[:2])
    #         for index in range(cur_index, len(model_structure)):
    #             layer_name = model_structure[str(index)]['args']['name']
    #             if has_weights(layer_name) and is_connected(index, cur_index):
    #                 return layer_name
    #         return None

    #     def params_delta(grads1, grads2, layer_name):
    #         g1_list = [g for n, g in grads1.items() if layer_name in n]
    #         g2_list = [g for n, g in grads2.items() if layer_name in n]
    #         return max([self.__max_delta(g1, g2) for g1, g2 in zip(g1_list, g2_list)])

    #     for layer_info in model_structure.values():
    #         layer_name = layer_info['args']['name']
    #         if not has_weights(layer_name):
    #             continue

    #         forward_layer_name = seek_forward_layer_name(layer_name)
    #         grads_delta = params_delta(grads1, grads2, layer_name)
    #         weights_delta = params_delta(weights1, weights2, layer_name)

    #         if forward_layer_name is None:  # 无forward层
    #             grads_delta_pre = 0
    #             weights_delta_pre = 0
    #         else:
    #             grads_delta_pre = params_delta(grads1, grads2, forward_layer_name)
    #             weights_delta_pre = params_delta(weights1, weights2, forward_layer_name)

    #         grads_R = (grads_delta - grads_delta_pre) / (grads_delta_pre + epsilon)
    #         weights_R = (weights_delta - weights_delta_pre) / (weights_delta_pre + epsilon)
    #         localization_map[layer_name] = (grads_delta, grads_R, weights_delta, weights_R, forward_layer_name)
    #     return localization_map
