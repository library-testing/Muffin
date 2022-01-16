from typing import List, Optional
import numpy as np
import random
from pathlib import Path

from utils.cmd_process import CmdProcess
from utils.utils import losses, optimizers


class Trainer(object):

    def __init__(self, db_manager, timeout):
        super().__init__()
        self.__db_manager = db_manager
        self.__timeout = timeout

    def train(self, model_id: int, exp_dir: str, ok_backends: List[str], loss: Optional[str] = None, optimizer: Optional[str] = None):
        '''把模型在每个ok_backend跑一次

        返回值：
            backends_weights以及ok_backend_list
        '''
        # 位置
        model_dir = Path(exp_dir) / 'models'
        training_inputs_path = Path(exp_dir) / 'dataset' / 'inputs.npz'
        ground_truths_path = Path(exp_dir) / 'dataset' / 'ground_truths.npz'

        outputs_dir = Path(exp_dir) / 'layer_outputs'
        loss_dir = Path(exp_dir) / 'loss'
        loss_grads_dir = Path(exp_dir) / 'loss_gradients'
        gradients_dir = Path(exp_dir) / 'layer_gradients'
        # weights_dir = Path(exp_dir) / 'layer_weights'

        outputs_dir.mkdir(parents=True, exist_ok=True)
        loss_dir.mkdir(parents=True, exist_ok=True)
        loss_grads_dir.mkdir(parents=True, exist_ok=True)
        gradients_dir.mkdir(parents=True, exist_ok=True)
        # weights_dir.mkdir(parents=True, exist_ok=True)

        # 随机选择losses和optimizers
        if loss is None:
            loss = random.choice(losses)
        if optimizer is None:
            optimizer = random.choice(optimizers)
        self.__db_manager.record_loss_optimizer(model_id, loss, optimizer)

        # 开始train
        crash_backends, nan_backends, inf_backends = [], [], []
        backends_outputs, backends_losses, backends_loss_grads, backends_grads, backends_weights = {}, {}, {}, {}, {}

        cmd_processes = {
            bk: CmdProcess(f"/root/anaconda3/envs/{bk}/bin/python -m src.incons_detection.train"
                           f" --backend {bk}"
                           f" --loss {loss}"
                           f" --optimizer {optimizer}"
                           f" --model_path {str(model_dir / f'{bk}.h5')}"
                           f" --model_info_path {str(model_dir / 'model.json')}"
                           f" --training_instances_path {str(training_inputs_path)}"
                           f" --ground_truths_path {str(ground_truths_path)}"
                           f" --outputs_dir {str(outputs_dir / bk)}"
                           f" --loss_path {str(loss_dir / f'{bk}.txt')}"
                           f" --loss_grads_dir {str(loss_grads_dir / bk)}"
                           f" --gradients_dir {str(gradients_dir / bk)}")
                           # f" --weights_dir {str(weights_dir / bk)}")
            for bk in ok_backends
        }

        status = {}
        for bk, p in cmd_processes.items():  # 改为顺序执行
            extract_status = p.run(self.__timeout)

            print(f"{bk}_status: {extract_status}")
            status[bk] = extract_status
            if extract_status and extract_status in [255, 1, -1]:  # 发生了output crash
                crash_backends.append(bk)

            else:
                outputs_data = loss_value = loss_grads_data = grads_data = None  # weights_data = None

                if extract_status == 0 or extract_status >= 2:
                    outputs_data = {fn.stem: np.load(str(fn)) for fn in (outputs_dir / bk).glob("*.npy")}
                if extract_status == 0 or extract_status >= 3:
                    with open(str(loss_dir / f'{bk}.txt'), 'r') as f:
                        loss_value = float(f.read())
                if extract_status == 0 or extract_status >= 4:
                    loss_grads_data = {fn.stem: np.load(str(fn)) for fn in (loss_grads_dir / bk).glob("*.npy")}
                if extract_status == 0 or extract_status >= 5:
                    grads_data = {fn.stem: np.load(str(fn)) for fn in (gradients_dir / bk).glob("*.npy")}
                # if extract_status == 0 or extract_status >= 6:
                #     weights_data = {fn.stem: np.load(str(fn)) for fn in (weights_dir / bk).glob("*.npy")}

                backends_outputs[bk] = outputs_data
                backends_losses[bk] = loss_value
                backends_loss_grads[bk] = loss_grads_data
                backends_grads[bk] = grads_data
                # backends_weights[bk] = weights_data

            if extract_status == 0 or (extract_status != 255 and extract_status >= 2):
                if self.__check(outputs_data, np.isnan):  # 如果存在nan
                    nan_backends.append(bk)
                if self.__check(outputs_data, np.isinf):  # 如果存在inf
                    inf_backends.append(bk)

        # 记录异常到数据库
        if crash_backends:  # 存在crash
            self.__db_manager.update_model_crash_backends(model_id, crash_backends)
        if nan_backends:  # 存在nan
            self.__db_manager.update_model_nan_backends(model_id, nan_backends)
        if inf_backends:  # 存在inf
            self.__db_manager.update_model_inf_backends(model_id, inf_backends)

        return status, backends_outputs, backends_losses, backends_loss_grads, backends_grads, [bk for bk in ok_backends if bk not in crash_backends]

    def __check(self, weights, f):
        for w in weights.values():
            if f(w).any():
                return True
        return False
