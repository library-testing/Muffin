from pathlib import Path
from typing import List, Optional
import datetime

from utils.db_manager import DbManager
from utils.utils import get_HH_mm_ss
from src.cases_generation.model_info_generator import ModelInfoGenerator
from src.cases_generation.model_generator import ModelGenerator
from src.cases_generation.data_generator import DataGenerator
from src.incons_detection.trainer import Trainer
from src.incons_detection.comparator import Comparator
from utils.selection import Roulette


class TrainingDebugger(object):

    def __init__(self, config: dict, use_heuristic: bool = True, generate_mode: str = 'template', timeout: float = 60):
        super().__init__()
        self.__output_dir = config['output_dir']
        self.__db_manager = DbManager(config['db_path'])
        # Roulette处理器
        from utils.utils import layer_types, layer_conditions
        self.__selector = Roulette(layer_types=layer_types,
                                   layer_conditions=layer_conditions,
                                   use_heuristic=use_heuristic)
        self.__model_info_generator = ModelInfoGenerator(config['model'], self.__db_manager, self.__selector, generate_mode)
        self.__model_generator = ModelGenerator(config['model']['var']['weight_value_range'], config['backends'], self.__db_manager, self.__selector, timeout)
        self.__training_data_generator = DataGenerator(config['training_data'])
        self.__weights_trainer = Trainer(self.__db_manager, timeout)
        self.__weights_comparator = Comparator(self.__db_manager)

    def run_generation(self, model_info: Optional[dict] = None, initial_weight_dir: Optional[str] = None, dataset_name: Optional[str] = None):
        '''随机生成模型和数据, 可指定现成模型信息
        '''
        # 随机生成model
        print('model生成开始...')
        json_path, model_input_shapes, model_output_shapes, model_id, exp_dir = self.__model_info_generator.generate(save_dir=self.__output_dir,
                                                                                                                     model_info=model_info)
        ok_backends = self.__model_generator.generate(json_path=json_path,
                                                      model_id=model_id,
                                                      exp_dir=exp_dir,
                                                      initial_weight_dir=initial_weight_dir)
        print(f'model生成完毕: model_id={model_id} ok_backends={ok_backends}')

        if len(ok_backends) >= 2:  # 否则没有继续实验的必要
            # 随机生成training data
            print('training data生成开始...')
            if dataset_name is None:
                self.__training_data_generator.generate(input_shapes=model_input_shapes,
                                                        output_shapes=model_output_shapes,
                                                        exp_dir=exp_dir)
            else:
                import shutil
                shutil.copytree(str(Path('dataset') / dataset_name), str(Path(exp_dir) / 'dataset'))
            print('training data生成完毕.')

        return model_id, exp_dir, ok_backends

    def run_generation_for_dataset(self, dataset_name: str):
        '''随机生成模型, 数据使用现成dataset
        '''
        # 随机生成model
        print('model生成开始...')
        json_path, _, _, model_id, exp_dir = self.__model_info_generator.generate_for_dataset(save_dir=self.__output_dir, dataset_name=dataset_name)
        ok_backends = self.__model_generator.generate(json_path=json_path,
                                                      model_id=model_id,
                                                      exp_dir=exp_dir)
        print(f'model生成完毕: model_id={model_id} ok_backends={ok_backends}')

        if len(ok_backends) >= 2:  # 否则没有继续实验的必要
            # 复制数据集
            print('training data生成开始...')
            import shutil
            shutil.copytree(str(Path('dataset') / dataset_name), str(Path(exp_dir) / 'dataset'))
            print('training data生成完毕.')

        return model_id, exp_dir, ok_backends

    def run_detection(self, model_id: int, exp_dir: str, ok_backends: List[str],  loss: Optional[str] = None, optimizer: Optional[str] = None):
        '''对一个model进行detection检测
        '''
        if len(ok_backends) >= 2:
            # Training阶段
            print('Training开始...')
            status, backends_outputs, backends_losses, backends_loss_grads, backends_grads, ok_backends = self.__weights_trainer.train(model_id=model_id,
                                                                                                                                       exp_dir=exp_dir,
                                                                                                                                       ok_backends=ok_backends,
                                                                                                                                       loss=loss,
                                                                                                                                       optimizer=optimizer)
            print(f'Training结束: ok_backends={ok_backends}')

            self.__db_manager.record_status(model_id, status)

        if len(ok_backends) >= 2:
            # Weights Comparator阶段
            print('Compare开始...')
            self.__weights_comparator.compare(model_id=model_id,
                                              exp_dir=exp_dir,
                                              backends_outputs=backends_outputs,
                                              backends_losses=backends_losses,
                                              backends_loss_grads=backends_loss_grads,
                                              backends_grads=backends_grads,
                                              # backends_weights=backends_weights,
                                              ok_backends=ok_backends)
            print('Compare结束.')

        return ok_backends

    def get_coverage(self):
        return self.__selector.coverage()


def main(testing_config):
    config = {
        'model': {
            'var': {
                'tensor_dimension_range': (2, 5),
                'tensor_element_size_range': (2, 5),
                'weight_value_range': (-10.0, 10.0),
                'small_value_range': (0, 1),
                'vocabulary_size': 1001,
            },
            'node_num_range': (5, 5),
            'dag_io_num_range': (1, 3),
            'dag_max_branch_num': 2,
            'cell_num': 3,
            'node_num_per_normal_cell': 10,
            'node_num_per_reduction_cell': 2,
        },
        'training_data': {
            'instance_num': 10,
            'element_val_range': (0, 100),
        },
        'db_path': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}.db'),
        'output_dir': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}_output'),
        'report_dir': str(Path.cwd() / testing_config['data_dir'] / f'{testing_config["dataset_name"]}_report'),
        'backends': ['tensorflow', 'theano', 'cntk'],
        'distance_threshold': 0,
    }

    DEBUG_MODE = testing_config["debug_mode"]
    CASE_NUM = testing_config["case_num"]
    TIMEOUT = testing_config["timeout"]  # 秒
    USE_HEURISTIC = bool(testing_config["use_heuristic"])  # 是否开启启发式规则
    GENERATE_MODE = testing_config["generate_mode"]  # seq\merging\dag\template

    debugger = TrainingDebugger(config, USE_HEURISTIC, GENERATE_MODE, TIMEOUT)
    start_time = datetime.datetime.now()

    if DEBUG_MODE == 1:  # 现成数据集 + 随机模型

        for i in range(CASE_NUM):
            print(f"######## Round {i} ########")
            try:
                print("------------- generation -------------")
                model_id, exp_dir, ok_backends = debugger.run_generation_for_dataset(testing_config["dataset_name"])
                print("------------- detection -------------")
                ok_backends = debugger.run_detection(model_id, exp_dir, ok_backends)
            except Exception:
                import traceback
                traceback.print_exc()

    # elif DEBUG_MODE == 2:  # 指定数据集 + 指定模型
    #     import json
    #     model_name = 'lenet5-fashion-mnist_origin0'
    #     model_info_path = Path("temp") / f"{model_name}.json"
    #     initial_weight_dir = str(Path("temp") / f"{model_name}_weights")
    #     loss = 'mean_absolute_error'
    #     optimizer = 'sgd'
    #     # dataset_name = 'mnist'

    #     with open(str(model_info_path), "r") as f:
    #         model_info = json.load(f)
    #     try:
    #         print("------------- generation -------------")
    #         model_id, exp_dir, ok_backends = debugger.run_generation(model_info=model_info, initial_weight_dir=initial_weight_dir, dataset_name=testing_config["dataset_name"])
    #         print("------------- detection -------------")
    #         ok_backends = debugger.run_detection(model_id, exp_dir, ok_backends, loss, optimizer)
    #     except Exception:
    #         import traceback
    #         traceback.print_exc()

    else:  # 随机数据集 + 随机模型
        for i in range(CASE_NUM):
            print(f"######## Round {i} ########")
            try:
                print("------------- generation -------------")
                model_id, exp_dir, ok_backends = debugger.run_generation()
                print("------------- detection -------------")
                ok_backends = debugger.run_detection(model_id, exp_dir, ok_backends)
            except Exception:
                import traceback
                traceback.print_exc()

    end_time = datetime.datetime.now()
    time_delta = end_time - start_time
    h, m, s = get_HH_mm_ss(time_delta)
    print(f"R-CRADLE is done: Time used: {h} hour,{m} min,{s} sec")
    # coverage_rate, selected_map = debugger.get_coverage()
    # print(f"Layer coverage is: {coverage_rate}")
    # with open("runnning_info.txt", "a") as f:
    #     print(f"{dataset_name} is done: Time used: {h} hour,{m} min,{s} sec", file=f)
    #     print(f"Layer coverage is: {coverage_rate}\n", file=f)


if __name__ == '__main__':
    import json
    with open(str("testing_config.json"), "r") as f:
        testing_config = json.load(f)
    main(testing_config)
    # datasets = ['cifar10', 'mnist', 'fashion_mnist', 'imagenet', 'sinewave', 'price']
    # for dataset in datasets:
    #     main(dataset, 'data', debug_mode=1)
