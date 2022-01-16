from typing import Tuple, Optional
from pathlib import Path
import numpy as np


class DataGenerator(object):

    def __init__(self, config: int):
        super().__init__()
        self.__instance_num = config['instance_num']
        self.__ele_val_range = config['element_val_range']

    def generate(self, input_shapes: dict, exp_dir: str, output_shapes: Optional[dict] = None):
        # 创建存放位置
        save_dir = Path(exp_dir) / 'dataset'
        save_dir.mkdir(parents=True, exist_ok=True)

        # 生成dataset
        data_inputs_path = save_dir / 'inputs.npz'
        data_inputs = {input_name: self.__generate(input_shape) for input_name, input_shape in input_shapes.items()}
        np.savez(data_inputs_path, **data_inputs)

        # 生成ground_truth
        if output_shapes:
            ground_truths_path = save_dir / 'ground_truths.npz'
            ground_truths = {output_name: self.__generate(output_shape) for output_name, output_shape in output_shapes.items()}
            np.savez(ground_truths_path, **ground_truths)

    def __generate(self, shape: Tuple[Optional[int]]):
        a, b = self.__ele_val_range
        return np.random.rand(*(self.__instance_num, *shape[1:])) * (b - a) + a


if __name__ == '__main__':
    config = {
        'instance_num': 5,
        'element_val_range': (-1000, 1000),
    }
    vig = DataGenerator(config)
    print(vig.generate((2, 3)))
