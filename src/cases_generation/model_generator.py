from typing import List, Optional
from pathlib import Path
import json

from utils.cmd_process import CmdProcess


class ModelGenerator(object):
    '''模型生成器
    '''

    def __init__(self, weight_val_range: tuple, backends: List[str], db_manager, selector, timeout):
        super().__init__()
        self.__weight_val_range = weight_val_range
        self.__backends = backends
        self.__db_manager = db_manager
        self.__selector = selector
        self.__timeout = timeout

    def generate(self, json_path: str, exp_dir: str, model_id: int, initial_weight_dir: Optional[str] = None):
        model_dir = Path(exp_dir) / 'models'

        # 用不同后端分别生成相同的model
        fail_backends = []
        first_bk = None

        if initial_weight_dir is None:
            # 先用一个后端生成模型，并保存其weights
            for bk in self.__backends:
                p = CmdProcess(f"/root/anaconda3/envs/{bk}/bin/python -m src.cases_generation.generate_one"
                               f" --backend {bk}"
                               f" --json_path {str(json_path)}"
                               f" --weight_minv {self.__weight_val_range[0]}"
                               f" --weight_maxv {self.__weight_val_range[1]}"
                               f" --output_dir {str(model_dir)}")
                generate_status = p.run(self.__timeout)

                if generate_status:  # 生成失败
                    fail_backends.append(bk)
                else:
                    first_bk = bk
                    break
        else:
            import shutil
            shutil.copytree(initial_weight_dir, str(Path(model_dir) / 'initial_weights'))

        # 剩下的模型可以并行生成，加载之前保存的weights
        cmd_processes = {
            bk: CmdProcess(f"/root/anaconda3/envs/{bk}/bin/python -m src.cases_generation.generate_other"
                           f" --backend {bk}"
                           f" --json_path {str(json_path)}"
                           f" --output_dir {str(model_dir)}")
            for bk in self.__backends if bk != first_bk and bk not in fail_backends
        }
        for bk, p in cmd_processes.items():
            generate_status = p.run(self.__timeout)

            if generate_status:  # 生成失败
                fail_backends.append(bk)
            else:
                self.__update_selected_layers_cnt(str(json_path))  # 更新layer的生成次数

        # 记录生成失败到数据库
        if fail_backends:  # 存在crash
            self.__db_manager.update_model_generate_fail_backends(model_id, fail_backends)

        return [bk for bk in self.__backends if bk not in fail_backends]

    def __update_selected_layers_cnt(self, json_path):
        with open(json_path, 'r') as f:
            model_info = json.load(f)
            for layer_info in model_info['model_structure'].values():
                self.__selector.update(name=layer_info['type'])


if __name__ == '__main__':
    pass
