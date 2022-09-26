import tensorflow as tf
import random as rn
import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['PYTHONHASHSEED'] = '0'

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


def seed_tensorflow(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_tensorflow(seed)
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
# from sklearn.externals import joblib
# import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
import tensorflow as tf

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Reshape

# fit an LSTM network to training data
def fit_lstm():
    x_input = Input(shape=(660, 4))
    x_input1 = LSTM(32,return_sequences=False)(x_input)
    x_input1=layers.Dropout(0.011)(x_input1)
    logits = layers.Dense(1)(x_input1)
    return Model(inputs=x_input, outputs=logits)

import tensorflow as tf
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 660, 4)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(0, len(dataset) - look_back, 2641):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    for i in range(len(dataY)):
        if dataY[i].astype("float64") == 0:
            dataY[i] = str(dataY[i - 1][0].astype("float64"))
    # 这里是一周期的数据dataX 和 容量 dataY
    return dataX, dataY


loss_list = []


def experiment(series5, series6, series7, series18, updates, look_back, neurons, n_epoch, batch_size):
    index = []
    raw_values5 = series5.values
    raw_values6 = series6.values
    raw_values7 = series7.values
    raw_values18 = series18.values

    dataset_5, dataY_5 = create_dataset(raw_values5, look_back)
    dataset_6, dataY_6 = create_dataset(raw_values6, look_back)
    dataset_7, dataY_7 = create_dataset(raw_values7, look_back)
    dataset_18, dataY_18 = create_dataset(raw_values18, look_back)

    # 划分训练集，测试集
    train_x = np.concatenate((dataset_5, dataset_6, dataset_7, dataset_18), axis=0)
    train_y = np.concatenate((dataY_5, dataY_6, dataY_7, dataY_18), axis=0)

    from sklearn.model_selection import cross_val_score, train_test_split, KFold
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=0)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_x)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)

    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(train_y)
    Y_train = scaler_y.transform(Y_train)
    Y_test = scaler_y.transform(Y_test)

    # X_train：(443, 660, 4)
    X_train = train_scaled.reshape(X_train.shape[0], 660, 4)
    X_train = np.array(X_train, dtype="float")
    y_train = np.array(Y_train, dtype="float")

    X_test = test_scaled.reshape(X_test.shape[0], 660, 4)
    X_test = np.array(X_test, dtype="float")
    y_test = np.array(Y_test, dtype="float")
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    from tensorflow.keras.callbacks import TensorBoard
    TensorBoard = TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, write_images=True)
    model = fit_lstm()
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                  metrics=['mse'])

    def get_flops(model):
        concrete = tf.function(lambda inputs: model(inputs))
        concrete_func = concrete.get_concrete_function(
            [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
            return flops.total_float_ops

    print("The FLOPs is:{}".format(get_flops(model)), flush=True)
    # In[24]:
    print(model.summary())
    # model.fit(X_train, y_train, batch_size=128, epochs=700, verbose=1)
    import time
    start = time.clock()
    history1 = model.fit(X_train, y_train, batch_size=128, epochs=300, verbose=1,callbacks=[TensorBoard])
    print('模型训练时间：', time.clock() - start, 's')
    global loss_list
    loss_list = history1.history["loss"]

    # todo:
    # 1、跑一下这个数据 done
    # 2、测试一下 label 进行归一化 done

    # 模型保存
    # model.save_weights('lstm.h5')  # 提供保存的路径
    # Restore the weights
    # model = fit_lstm()  # 重新创建网络
    # model.load_weights('lstm.h5')
    def MAPE(true, pred):
        diff = np.abs(np.array(true) - np.array(pred))
        return np.mean(diff / true)
    # model = fit_lstm()  # 重新创建网络
    # model.load_weights('best.hdf5')
    print('#'*100)
    print(model)
    print('#' * 100)
    y_pred = model.predict(X_test)
    y_true = y_test
    rmse_test = sqrt(mean_squared_error(y_true, y_pred))
    mae_test = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    print('rmse:', rmse_test)
    print('mae:', mae_test)
    print('r2:', R2)
    print('MAPE:', mape)
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.show()
    # In[25]:
    # print('预测测试集数据')
    # predictions_test = list()
    # for i in range(len(X_test)):
    #     # 进行单步预测
    #     yhat = forecast_lstm(model, batch_size, X_test[i])
    #     predictions_test.append(yhat)  # 存储预测数据
    #     expected = y_test[i]
    #     print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
    # # 计算RMSE
    # rmse_test = sqrt(mean_squared_error(y_test.astype("float64"), predictions_test))
    # mae_test = mean_absolute_error(y_test.astype("float64"), predictions_test)
    # R2 = r2_score(y_test.astype("float64"), predictions_test)
    # print('测试集 RMSE: %.3f' % rmse_test)
    # print('测试集 MAE: %.3f' % mae_test)
    # print('测试集 R2_SCORE: %.3f' % R2)
    # index.append(rmse_test)

from memory_profiler import profile
@profile
def run():
    global file

    file_name1 = './data/vltm5.csv'
    file_name2 = './data/vltm6.csv'
    file_name3 = './data/vltm7.csv'
    file_name4 = './data/vltm18.csv'




    file = file_name1[7:-4]
    series1 = read_csv(file_name1, header=None, parse_dates=[0], squeeze=True, sep=',')
    series2 = read_csv(file_name2, header=None, parse_dates=[0], squeeze=True, sep=',')
    series3 = read_csv(file_name3, header=None, parse_dates=[0], squeeze=True, sep=',')
    series4 = read_csv(file_name4, header=None, parse_dates=[0], squeeze=True, sep=',')

    look_back = 2640
    neurons = [64, 64]
    n_epochs = 2  # 252
    updates = 1
    batch_size = 26
    experiment(series1, series2, series3, series4, updates, look_back, neurons, n_epochs, batch_size)


run()
with open("./loss/lstm-loss.csv", "w", encoding="utf-8") as fout:
	for losss_data in zip(loss_list):
		fout.writelines(str(losss_data[0])+ "\n")
fig = plt.figure()
plt.plot(loss_list, label='loss', color='blue')
plt.legend()
plt.title('model loss')
plt.savefig('./result/loss-lstm-' + file + ".png")
plt.show()
