import sys
import os
import json
import numpy as np
from pathlib import Path

os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras

fn = sys.argv[1]

model = keras.models.load_model(fn)

# 保存图片
pic_path = f'{fn[:-3]}.png'
keras.utils.plot_model(model, str(pic_path), show_shapes=True)

model_structure = dict()

model_structure["0"] = {
    "type": "input_object",
    "args": {
        "shape": [
            28,
            28,
            1
        ],
        "batch_shape": None,
        "dtype": None,
        "sparse": False,
        "tensor": None,
        "name": "00_input_object"
    },
    "pre_layers": [],
    "output_shape": [
        None,
        28,
        28,
        1
    ]
}

for index, layer in enumerate(model.layers, start=1):
    type_name = layer.__class__.__name__
    args = layer.get_config()
    args["name"] = f"{str(index).zfill(2)}_{type_name}"
    layer_output = [(i if i is None else int(i)) for i in layer.output_shape]
    model_structure[str(index)] = {
            "type": type_name,
            "args": args,
            "pre_layers": [index-1],
            "output_shape": layer_output
        }

model_info = {
    "model_structure": model_structure,
    "input_id_list": [0],
    "output_id_list": [len(model.layers)],
    "dataset_name": "mnist"
}


with open(str(f'{fn[:-3]}.json'), 'w') as f:
    json.dump(model_info, f)


weights_dir = Path(f"{fn[:-3]}_weights")
weights_dir.mkdir(parents=True, exist_ok=True)
save_path = weights_dir / '00_input_object.npz'
np.savez(save_path, *[])
for index, layer in enumerate(model.layers, start=1):
    type_name = layer.__class__.__name__
    layer_name = f"{str(index).zfill(2)}_{type_name}"
    save_path = weights_dir / f'{layer_name}.npz'
    np.savez(save_path, *layer.get_weights())
