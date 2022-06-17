# Muffin: Testing Deep Learning Libraries via Neural Architecture Fuzzing


This is the implementation repository of our *ICSE'22* paper: **Muffin: Testing Deep Learning Libraries via Neural Architecture Fuzzing**.



## Description

Deep learning (DL) techniques are shown to be effective in many challenging tasks, and are hence widely-adopted in practice. However, previous work has shown that DL libraries, the basis of building and executing DL models, contain bugs and can cause severe consequences. Unfortunately, existing approaches still cannot comprehensively excise DL libraries. They utilize existing trained models and only detect bugs in model inference phase. In this work we propose ***Muffin*** to address these issues. To this end, *Muffin* applies a specifically-designed model fuzzing approach, which allows it to generate diverse DL models to explore the target library, instead of relying only on existing trained models. *Muffin* makes differential testing feasible in the model training phase by tailoring a set of metrics to measure the inconsistency between different DL libraries. In this way, *Muffin* can best excise the library code to detect more bugs. Experiments on three widely-used DL libraries show that *Muffin* can detect **39 new bugs**.

You can access this repository using the following command:
```shell
git clone https://github.com/library-testing/Muffin.git
```



## Libraries

We use three widely-used DL libraries (*i.e.*, ***TensorFlow***, ***Theano***, and ***CNTK***) as the backend low-level libraries as our testing targets, and ***Keras*** as the frontend high-level library. To sufficiently illustrate the effectiveness of ***Muffin***, we utilize a total of **15 release versions** of the three backend libraries, and construct five experimental environments for differential testing as follow:

| ID   | Keras | TensorFlow | Theano | CNTK  |
| ---- | ----- | ---------- | ------ | ----- |
| E1   | 2.3.1 | 2.0.0      | 1.0.4  | 2.7.0 |
| E2   | 2.3.1 | 1.15.0     | 1.0.3  | 2.6.0 |
| E3   | 2.2.4 | 1.12.0     | 1.0.2  | 2.5.0 |
| E4   | 2.2.4 | 1.11.0     | 1.0.1  | 2.4.0 |
| E5   | 2.2.4 | 1.10.0     | 1.0.0  | 2.3.0 |

In order to facilitate other researchers to reproduce ***Muffin***, we provide ***docker*** images for each experiments (*i.e.*, E1 ~ E5), named `librarytesting/muffin` with tags from `E1` to `E5` respectively.

If you don't want to reproduce the experiments, you can directly get the output of E1 in [E1_output.zip](https://drive.google.com/file/d/1_dI0UjHKYosPkrIVC6kQMTB3GkCQHHBc/view?usp=sharing).



## Datasets(Optional)

Our approach is not sensitive to datasets, *i.e.*, theoretically any data type can be used for testing.  So you can just test with **randomly generated dataset** with our source code, for briefness.  

If you want to do comparative experiments with existing approaches, 6 widely-used datasets mentioned in our paper can be used, *i.e.*, **MNIST**, **F-MNIST**, **CIFAR-10**, **ImageNet**, **Sine-Wave** and **Stock-Price**. The first three ones can be accessed by [Keras API](https://keras.io/api/datasets/)，while the rest can be access from [OneDrive](https://onedrive.live.com/?authkey=%21ANVR8C2wSN1Rb9M&id=34CB15091B189D3E%211909&cid=34CB15091B189D3E)(`dataset.zip`) provided by [LEMON](https://github.com/Jacob-yen/LEMON).



## Environment

As we mentioned above, ***docker*** images are provided for experiments.

* **Step 0:** Please install ***nvidia-docker2***. You can install it by following this [document](https://codepyre.com/2019/01/installing-nvidia-docker2-on-ubuntu-18.0.4/).
* **Step 1:** Clone this repository into `[PATH_TO_MUFFIN]`. If you want to use the last three dataset mentioned above, download and unzip them into `[PATH_TO_MUFFIN]/dataset`.
   
   `[PATH_TO_MUFFIN]` is the local path you want to `git clone` this repository into.

* **Step 2:** Use the following command to pull the ***docker*** image we released (take `E1` as an example), and create a container for it:

  ```shell
  docker pull librarytesting/muffin:E1
  docker run --runtime=nvidia -it -v [PATH_TO_MUFFIN]:/data --name muffin-E1 librarytesting/muffin:E1 /bin/bash
  ```

  At this point, you're inside the ***docker*** container and ready to the experiment.

* **Step 3**: Enter the virtual environment we have set up in the container:

  ```shell
  source activate lemon
  ```

  

## Experiments

Make sure you are now in the ***docker*** container!

#### 1. Configuration

A configuration file `testing_config.json` should be provided to flexibly set up testing configuration. Here is an example:

```json
{
    "debug_mode": 1,
    "dataset_name": "mnist",
    "case_num": 200,
    "generate_mode": "seq",
    "data_dir": "data",
    "timeout": 300,
    "use_heuristic": 1
}
```

* `debug_mode` can be set between `0` and `1`.  `0`  represents testing with **randomly generated dataset**, `1` represents testing with existing datasets we mentioned above. In mode `0`, no dataset is required.
* `dataset_name` indicates the name of dataset in mode `1`. The available options include `cifar10`, `mnist`, `fashion_mnist`, `imagenet`, `sinewave`, `price`. If you test in mode `0`, you **should** randomly set a name that does not conflict with the existing name for distinction.
* `case_num` indicates the number of random models to generate for test.
* `generate_mode` indicates the mode of model generation,. The available options include  `seq`, `merging`, `dag`, `template`.
* `data_dir` indicates the data directory to store outputs.  Remaining `data` is recommended.
* `timeout` indicates the timeout duration for each model.
* `use_heuristic` indicates if using the heuristic method mentioned in the paper or not. `1` is recommended.

#### 2. Preprocessing

* **Dataset:** Execute the following command in `/data/dataset` to preprocess the dataset or downloading them from [Keras API](https://keras.io/api/datasets/) if you want to use existing dataset:

	```shell
	python get_dataset.py [DATASET_NAME]
	```

	`[DATASET_NAME]` can be chosen from `cifar10`、 `mnist`、`fashion_mnist`、`imagenet`、`sinewave`、`price`.
	
* **Generate database:** Execute the following command in `/data/data` to create the ***sqlite*** database for storing the testing results:

  ```
  sqlite3 [DATASET_NAME].db < create_db.sql
  ```

  Remember to change `[DATASET_NAME]` to the name of the dataset you have set in `testing_config.json`.

#### 3. Start

Use the following command to run the experiment according to the configuration:

```shell
python run.py
```

The `testing_config.json` file should be place in the same directory.

The testing results will be store in `/data/[DATA_DIR]/[DATASET_NAME].db` (*e.g.* `/data/data/mnist.db`), and **detail results** for each model will be stored in `/data/[DATA_DIR]/[DATASET_NAME]_output`. 

Use the following command in `/data/data` to delete a set of testing results ( **carefully use!** ):

```shell
python clear_data.py [DATASET_NAME]
```


## Citation

```
@inproceedings{gu2022muffin,
  title={Muffin: Testing Deep Learning Libraries via Neural Architecture Fuzzing},
  author={Gu Jiazhen, Luo Xuchuan, Zhou Yangfan, Wang Xin},
  booktitle={International Conference on Software Engineering (ICSE’22)},
  year={2022}
}
```

