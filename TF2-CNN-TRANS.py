import pandas as pd
import tensorflow as tf
import random as rn
import numpy as np
import os
import time

from tensorboard.plugins import projector

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['PYTHONHASHSEED'] = '0'

# fix random seed for reproducibility
# seed = 42
# np.random.seed(seed)


# def seed_tensorflow(seed=42):
#     np.random.seed(seed)
#     tf.random.set_seed(seed)


# seed_tensorflow(seed)
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

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
# from memory_profiler import profile
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Reshape
# from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import TensorBoard
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# fit an LSTM network to training data
def fit_lstm():
    num_heads = 14
    projection_dim = 11
    x_input1 = Input(shape=(660, 3))
    # x_input1 = layers.Conv1D(filters=34, kernel_size=1, padding='same', activation='relu')(x_input)
    # x_input1 = layers.MaxPool1D(pool_size=7, padding='valid')(x_input1)
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.0799
    )(x_input1, x_input1)
    # Skip connection 1.
    x2 = layers.Add()([attention_output, x_input1])
    # Layer normalization 2.
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP.
    x3 = mlp(x3, hidden_units=[3], dropout_rate=0.0001)

    # Skip connection 2.
    encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.0055)(representation)
    # Classify outputs.
    logits = layers.Dense(1)(representation)
    logits = layers.Dropout(0.0055)(logits)
    # Create the Keras model.
    return Model(inputs=x_input1, outputs=logits)


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 660, 4)
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # from sklearn.decomposition import PCA, KernelPCA
    # import pandas as pd
    # import numpy as np
    # df_iris = pd.DataFrame(data=dataset)
    # mle_pca = PCA(n_components='mle', svd_solver='full',)
    # X_pca = mle_pca.fit_transform(df_iris)
    # dataset = X_pca.reshape(-1, )
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
    all_data = np.concatenate((raw_values5, raw_values6, raw_values7, raw_values18), axis=0)
    # all_data = all_data.reshape(-1, 4)
    # print('all_data.shape',all_data.shape)
    # print(raw_values5.shape)
    print("#"*100)
    print(all_data.shape)

    # p
    train_x, train_y = create_dataset(all_data, look_back)

    from sklearn.decomposition import PCA, KernelPCA
    mle_pca = PCA(n_components='mle', svd_solver='full', )
    train_x = np.array(train_x).reshape(-1,4)
    print('PCA之前',train_x.shape)

    train_x = mle_pca.fit_transform(train_x)

    train_x = np.array(train_x).reshape(train_y.shape[0], -1)
    print('PCA之后', train_x.shape)
    from sklearn.model_selection import cross_val_score, train_test_split, KFold
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=0)
    print('X_train.shape', np.array(X_train).shape)
    print('X_test.shape', np.array(X_test).shape)
    print('Y_train.shape', np.array(Y_train).shape)
    print('Y_test.shape', np.array(Y_test).shape)
    #MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
    scaler = MinMaxScaler()
    scaler = scaler.fit(train_x)
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)

    scaler_y = MinMaxScaler()
    k = scaler_y.fit(train_y)
    Y_train = k.transform(Y_train)
    Y_test_trans = k.transform(Y_test)
    # o = k.inverse_transform(Y_test)
    # print(o)
    # print('Y_test.shape',Y_test.shape)
    # X_train：(443, 660, 4)
    X_train = train_scaled.reshape(-1, 660, 3)
    X_train = np.array(X_train, dtype="float")
    y_train = np.array(Y_train, dtype="float")

    X_test = test_scaled.reshape(-1, 660, 3)
    X_test = np.array(X_test, dtype="float")
    y_test = np.array(Y_test_trans, dtype="float")
    # from thop import profile
    # from torchstat import stat



    model = fit_lstm()
    # stat(model, (1,660,3))
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

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
    print(model.summary())
    # model.layers[-1].get_weights()
    # Model(inputs=model.input,)
    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(lr=0.00055, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0,
                                                     amsgrad=False),
                  metrics=['mse'])
    # In[24]:
    # from tensorflow.keras.callbacks import TensorBoard
    # TensorBoard = TensorBoard(log_dir='logs',write_graph=True,histogram_freq=1,write_images=True)
    # model.fit(X_train, y_train, batch_size=128, epochs=700, verbose=1)
    checkpoint_path_best = "best.hdf5"
    modelcheckpoint_best = keras.callbacks.ModelCheckpoint(checkpoint_path_best,
                                  monitor='loss',
                                  save_best_only=True,
                                  mode='min',verbose=1)
    import time
    start = time.clock()
    # %memit
    history1 = model.fit(X_train, y_train, batch_size=16, epochs=30,verbose=0,callbacks=[modelcheckpoint_best])
    print('模型训练时间：',time.clock() - start,'s')
    global loss_list
    loss_list = history1.history["loss"]
    # model = model.load_weights('best.hdf5')
    # flops, params = profile(model, inputs=(32, 660, 3), verbose=True)
    # print("%s | %.2f | %.2f" % (model, params / (1000 ** 2), flops / (1000 ** 3)))  # 转化为MB

    # todo:
    # 1、跑一下这个数据 done
    # 2、测试一下 label 进行归一化 done

    # 模型保存
#     model.save_weights('./checkpoints/my_checkpoint_only_attention')  # 提供保存的路径
    # Restore the weights
    def MAPE(true, pred):
        diff = np.abs(np.array(true) - np.array(pred))
        return np.mean(diff / true)
    model = fit_lstm()  # 重新创建网络
    model.load_weights('best.hdf5')
    ##########################################
    #########################################
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
    print('mape:', mape)
    # y_true = y_test.reshape(Y_test.shape[0],Y_test.shape[1])
    # y_true = k.inverse_transform(y_true)
    # y_pred = k.inverse_transform(y_pred)
    scaler_test = MinMaxScaler().fit(Y_test)
    y_true = scaler_test.inverse_transform(y_true)

    y_pred = scaler_test.inverse_transform(y_pred)
    pd.DataFrame(y_true).to_csv('y_true.csv')
    pd.DataFrame(y_pred).to_csv('y_pred.csv')
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.show()
    # In[25]:
#     print('预测测试集数据')
#     predictions_test = list()
#     for i in range(len(X_test)):
#         # 进行单步预测
#         yhat = forecast_lstm(model, batch_size, X_test[i])
#         predictions_test.append(yhat)  # 存储预测数据
#         expected = y_test[i]
#         print('Cycle=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))
#     # 计算RMSE
#     rmse_test = sqrt(mean_squared_error(y_test.astype("float64"), predictions_test))
#     mae_test = mean_absolute_error(y_test.astype("float64"), predictions_test)
#     R2 = r2_score(y_test.astype("float64"), predictions_test)
#     print('测试集 RMSE: %.3f' % rmse_test)
#     print('测试集 MAE: %.3f' % mae_test)
#     print('测试集 R2_SCORE: %.3f' % R2)
#     index.append(rmse_test)

# from memory_profiler import profile
# @profile
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
with open("./loss/cnn-transformer-loss.csv", "w", encoding="utf-8") as fout:
	for losss_data in zip(loss_list):
		fout.writelines(str(losss_data[0])+ "\n")
fig = plt.figure()
plt.plot(loss_list, label='loss', color='blue')
plt.legend()
plt.title('model loss')
plt.savefig('./result/loss-train_only_attention-' + file + ".png")
plt.show()
