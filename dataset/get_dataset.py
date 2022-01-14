import numpy as np
from pathlib import Path
from PIL import Image
import sys
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras


dataset_name = sys.argv[1] # e.g. 'fashion_mnist'


if dataset_name == 'cifar10':
    def get_cifar10_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 32, 32
        x_test = x_test.reshape(x_test.shape[0], w, h, 3)
        return x_test
    _, (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test = get_cifar10_data(x_test)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)


if dataset_name == 'mnist':
    def get_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        return x_test
    _, (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = get_mnist_data(x_test)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)


if dataset_name == 'fashion_mnist':
    def get_fashion_mnist_data(x_test):
        x_test = x_test.astype('float32') / 255.0
        w, h = 28, 28
        x_test = x_test.reshape(x_test.shape[0], w, h, 1)
        return x_test
    _, (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test = get_fashion_mnist_data(x_test)
    y_test = keras.utils.to_categorical(y_test, num_classes=10)


if dataset_name == 'imagenet':
    def image_resize(x, shape):
        x_return = []
        for x_test in x:
            tmp = np.copy(x_test)
            img = Image.fromarray(tmp.astype('uint8')).convert('RGB')
            img = img.resize(shape, Image.ANTIALIAS)
            x_return.append(np.array(img))
        return np.array(x_return)
    data = np.load("sampled_imagenet-1500.npz")
    x, y = data['x_test'], data['y_test']
    shape = (224, 224)
    x_resize = image_resize(np.copy(x), shape)
    x_test = keras.applications.imagenet_utils.preprocess_input(x_resize)
    y_test = keras.utils.to_categorical(y, num_classes=1000)


if dataset_name == 'sinewave':
    import pandas as pd
    dataframe = pd.read_csv("./sinewave.csv")
    test_size, seq_len = 1500, 50
    data_test = dataframe.get("sinewave").values[-(test_size + 50):]
    data_windows = []
    for i in range(test_size):
        data_windows.append(data_test[i:i + seq_len])
    data_windows = np.array(data_windows).astype(float).reshape((test_size, seq_len, 1))
    data_windows = np.array(data_windows).astype(float)
    x_test = data_windows[:, :-1]
    y_test = data_windows[:, -1, [0]]


if dataset_name == 'price':
    def get_price_data():
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler

        def create_dataset(dataset, look_back=1):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i:(i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

        df = pd.read_csv("DIS.csv", header=None, index_col=None, delimiter=',')
        all_y = df[5].values
        dataset = all_y.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.5)
        # test_size = len(dataset) - train_size
        train, _ = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1, timestep 240
        look_back = 240
        trainX, trainY = create_dataset(train, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        return trainX, trainY
    x_test, y_test = get_price_data()


instance_num = 10

dataset_dir = Path(".") / dataset_name
dataset_dir.mkdir(exist_ok=True)
np.savez(str(dataset_dir / "inputs.npz"), x_test[:instance_num])
np.savez(str(dataset_dir / "ground_truths.npz"), y_test[:instance_num])

print(x_test[:instance_num].shape)
print(y_test[:instance_num].shape)
