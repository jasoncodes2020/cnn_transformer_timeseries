import os
import time
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from keras import backend as K
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout,LSTM
from recurrent import ATSLSTM
from keras.utils import plot_model


seed = 7
np.random.seed(seed)


loss_list = []

# scale train and test data to [-1, 1]
def scale(train, test):
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


# fit an AST-LSTM network to training data
def fit_lstm(X,y, batch_size, nb_epoch, neurons):
	# X, y = train[:, 0:-1], train[:, -1]
	# X = X.reshape(X.shape[0], 660, 4)
	# print(X.shape)
	# print(y.shape)
	model = Sequential()
# 	model.add(Conv1D(filters=46, kernel_size=7, strides=4, padding='same', activation='relu', input_shape=(X.shape[1], X.shape[2])))
# 	model.add(MaxPooling1D(pool_size=2, padding='valid'))
	model.add(ATSLSTM(24, return_sequences=True))
	model.add(ATSLSTM(28, return_sequences=False))
	model.add(Dropout(0.0609))
	model.add(Dense(1))
# 	adam = optimizers.Adam(lr=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# 	model.compile(loss='mean_squared_error', optimizer=adam)
# 	for i in range(nb_epoch):
# 		print('Epoch:',i)
# 		history = model.fit(X, y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
# 		loss_list.append(history.history['loss'][0])
# 		with open('./result/soh_loss.txt', 'a', encoding='utf-8') as f:
# 			f.write(str(history.history['loss'][0]) + "\n")
# 		model.reset_states()
# 		# plot_model(model, to_file=r'./result/soh_model_structure.png', show_shapes=True)
# 	model.save(r'./result/soh_model.h5')
	return model


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
    X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

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

    model = fit_lstm(X_train,y_train, batch_size, n_epoch, neurons)
    from keras.callbacks import TensorBoard
    TensorBoard = TensorBoard(log_dir='logs', write_graph=True, histogram_freq=1, write_images=True)
    model.compile(loss='mse',
                  optimizer=keras.optimizers.Adam(lr=0.00055, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0,
                                                     amsgrad=False),
                  metrics=['mse'])
    # In[24]:

    # model.fit(X_train, y_train, batch_size=128, epochs=700, verbose=1)
    history1 = model.fit(X_train, y_train, batch_size=512, epochs=300, verbose=1,validation_split=0.2,callbacks=[TensorBoard])
    global loss_list
    loss_list = history1.history["loss"]

    # todo:
    # 1、跑一下这个数据 done
    # 2、测试一下 label 进行归一化 done

    # 模型保存
    model.save_weights('./checkpoints/my_checkpoint_only_attention')  # 提供保存的路径
    # Restore the weights
    # model = fit_lstm(train_scaled, batch_size, n_epoch, neurons)  # 重新创建网络
    model.load_weights('./checkpoints/my_checkpoint_only_attention')
    y_pred = model.predict(X_test)
    y_true = y_test
    rmse_test = sqrt(mean_squared_error(y_true, y_pred))
    mae_test = mean_absolute_error(y_true, y_pred)
    R2 = r2_score(y_true, y_pred)
    print('rmse:', rmse_test)
    print('mae:', mae_test)
    print('r2:', R2)
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
    batch_size = 512
    experiment(series1, series2, series3, series4, updates, look_back, neurons, n_epochs, batch_size)


run()
# with open("./loss/cnn-transformer-loss.csv", "w", encoding="utf-8") as fout:
# 	for losss_data in zip(loss_list):
# 		fout.writelines(str(losss_data[0])+ "\n")
fig = plt.figure()
plt.plot(loss_list, label='loss', color='blue')
plt.legend()
plt.title('model loss')
plt.savefig('./result/loss-train_only_attention-' + file + ".png")
plt.show()


