
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,MaxAbsScaler,RobustScaler
import matplotlib.pyplot as plt
loss_list = []
# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 660, 4)
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	# print(dataset.shape)
	# from sklearn.decomposition import PCA, KernelPCA
	# import pandas as pd
	# import numpy as np
	# df_iris = pd.DataFrame(data=dataset)
	# mle_pca = PCA(n_components='mle', svd_solver='full')
	# X_pca = mle_pca.fit_transform(df_iris)
	# dataset = X_pca.reshape(-1,)
	dataX, dataY = [], []
	for i in range(0, len(dataset)-look_back, 2641):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	dataY= np.array(dataY)
	# print(dataX.shape)
	# print(dataY.shape)
	dataY = np.reshape(dataY,(dataY.shape[0],1))
	for i in range(len(dataY)):
		if dataY[i].astype("float64") == 0:
			dataY[i] = str(dataY[i-1][0].astype("float64"))
	# 这里是一周期的数据dataX 和 容量 dataY
	return dataX, dataY
from torch.utils.data import Dataset
class StockDataset(Dataset):
    def __init__(self, X_train, X_test, Y_train, Y_test, is_test=False):
        if not is_test:
            self.data = X_train
            self.label = Y_train
        else:
            self.data = X_test
            self.label = Y_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).to(torch.float32), torch.FloatTensor([self.label[idx]])
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn import preprocessing
import numpy as np
# from torchstat import stat
# from torchinfo import summary
from torchstat import stat
from attention import CNNLSTMModel_SE, CNNLSTMModel,CNNLSTMModel_HW,CNNLSTMModel_CBAM,CNNLSTMModel_ECA
import torch.nn as nn
import torchstat
# from sklearn.decomposition import
from tensorboardX import SummaryWriter
loss_list = []

def experiment(series5, series6, series7, series18, updates,look_back,neurons,n_epoch, batch_size):
	index = []
	raw_values5 = series5.values
	raw_values6 = series6.values
	raw_values7 = series7.values
	raw_values18 = series18.values
	all_data = np.concatenate((raw_values5, raw_values6, raw_values7, raw_values18), axis=0)
	# all_data = all_data.reshape(-1,4)
	# print('all_data.shape',all_data.shape)
	# print(raw_values5.shape)
	# print(train_x.shape)

	# p
	train_x,train_y = create_dataset(all_data, look_back)
	# dataset_5, dataY_5 = create_dataset(raw_values5, look_back)
	# dataset_6, dataY_6 = create_dataset(raw_values6, look_back)
	# dataset_7, dataY_7 = create_dataset(raw_values7, look_back)
	# dataset_18, dataY_18 = create_dataset(raw_values18, look_back)

	# 划分训练集，测试集
	# train_x = np.concatenate((dataset_5, dataset_6, dataset_7, dataset_18), axis=0)
	# print('train_x.shape',train_x.shape)
	# # p
	# train_y = np.concatenate((dataY_5, dataY_6, dataY_7, dataY_18), axis=0)

	from sklearn.model_selection import cross_val_score, train_test_split, KFold
	X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.4, random_state=0)
	#MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
	# scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = MinMaxScaler()
	scaler = scaler.fit(train_x)
	train_scaled = scaler.transform(X_train)
	print('train_scaled.shape',train_scaled.shape)
	# t
	test_scaled = scaler.transform(X_test)
	scaler_y = MinMaxScaler().fit(train_y)
	# scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(train_y)
	Y_train = scaler_y.transform(Y_train)
	Y_test = scaler_y.transform(Y_test)

	# X_train：(443, 660, 3)
	X_train= train_scaled.reshape(-1, 660, 4)
	X_train= np.array(X_train, dtype="float")
	y_train= np.array(Y_train, dtype="float").reshape(-1,)

	X_test = test_scaled.reshape(-1, 660, 4)
	X_test = np.array(X_test, dtype="float")
	y_test = np.array(Y_test, dtype="float")
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	print_step = 10
	train_data = StockDataset(X_train, X_test, y_train, y_test, is_test=False)
	test_data = StockDataset(X_train, X_test, y_train, y_test, is_test=True)
	train_loader = DataLoader(train_data, batch_size=128, shuffle=False, num_workers=0)
	test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)
	model = CNNLSTMModel()
	dummy_input = torch.randn(16, 1, 4)
	# flops, params = profile(model, (X_train,))
	# print('flops: ', flops, 'params: ', params)
	# torchstat(model,(660,4))
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# model = model.to(device)
	# summary(model, input_size=(128,660,4))
	# stat(model,(16, 4, 1))
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
	train_losses = []
	min_loss = 99999
	import time
	start = time.clock()
	writer = SummaryWriter(log_dir='logs')
	for epoch in range(30):
		# print(f'epoch:{epoch}')
		running_loss = 0.0
		# train_loss = 0
		for step, (data, label) in enumerate(train_loader):
			# flops, params = profile(model, (train_loader,))
			# print('flops: ', flops, 'params: ', params)
			out = model(data)
			loss = criterion(out, label)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			mse_loss = 0.0
			mse_loss += loss.item()
			torch.save(model.state_dict(), f"CNN_LSTM_model.pth")
			# if mse_loss / len(train_loader) < min_loss:
				# min_loss = mse_loss / len(train_loader)
				# print('更新最好模型')
				# torch.save(model.state_dict(), f"best_model/best_model.pth")
		print(f"epoch:{epoch}, train loss:{running_loss / len(train_loader)}")
		# torch.save(model.state_dict(), f"save_model/SE_model.pth")
		train_losses.append(running_loss / len(train_loader))
		# print(f"epoch:{epoch}, train loss:{train_losses}")
		writer.add_scalar('Train_loss',running_loss/(epoch+1),epoch)
		for i, (name, param) in enumerate(model.named_parameters()):
			writer.add_histogram(name, param, 0)
			# writer.add_d
		# writer.add_scalar('loss', loss[i], i)
	print('模型训练时间：', time.clock() - start, 's')
	pd.DataFrame(train_losses).to_csv('cnn-lstm-loss.csv')
	plt.plot(train_losses[2:])
	plt.title("loss曲线展示", fontsize='15')  # 添加标题
	plt.xlabel('epoch', fontsize='15')
	plt.ylabel('loss', fontsize='15')
	plt.show()
	plt.savefig("loss.jpg")
	params = torch.load(f"CNN_LSTM_model.pth")
	model.load_state_dict(params)

	eval_loss = 0.0
	with torch.no_grad():
		y_gt = []
		y_pred = []
		for data, label in test_loader:
			# label = label.squeeze(axis=1)
			# print(label.shape)
			y_gt += label.squeeze(axis=1).tolist()
			out = model(data)
			# print('data.shape',data.shape)
			# print('out.shape', out.shape)
			loss = criterion(out, label)
			eval_loss += loss.item()
			y_pred += out.numpy().squeeze(axis=1).tolist()
		print(len(y_gt), len(y_pred))
		print(np.array(y_gt).shape, np.array(y_pred).shape)
	# plt.plot(y_gt, color='r')
	# plt.plot(y_pred, color='b')
	# plt.show()
	# y_gt = np.array(y_gt)
	# y_gt = y_gt[:, np.newaxis]
	# y_pred = np.array(y_pred)
	# y_pred = y_pred[:, np.newaxis]
	def MAPE(true, pred):

		diff = np.abs(np.array(true) - np.array(pred))
		true = np.array(true)
		for i in range(len(true)):
			if true[i]==0:
				print('为零索引：',i)
				true[i]+=0.00001
		return np.mean(diff / true)
	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
	from numpy import sqrt
	y_gt = np.array(y_gt).reshape(-1,1)
	y_pred = np.array(y_pred).reshape(-1, 1)
	mape = MAPE(y_gt, y_pred)
	print('RMSE:', sqrt(mean_squared_error(y_gt, y_pred)))
	print('MAE:', mean_absolute_error(y_gt, y_pred))
	print('r2_score:', r2_score(y_gt, y_pred))
	print('MAPE:', mape)
from memory_profiler import profile
@profile
def run():
	global file
	file_name1 = './data/vltm5.csv'
	file_name2 = './data/vltm6.csv'
	file_name3 = './data/vltm7.csv'
	file_name4 = './data/vltm18.csv'



	file = file_name1[7:-4]
	series1 = read_csv(file_name1, header=None, parse_dates=[0], squeeze=True,sep=',')
	series2 = read_csv(file_name2, header=None, parse_dates=[0], squeeze=True,sep=',')
	series3 = read_csv(file_name3, header=None, parse_dates=[0], squeeze=True,sep=',')
	series4 = read_csv(file_name4, header=None, parse_dates=[0], squeeze=True,sep=',')

	look_back = 2640
	neurons = [64, 64]
	n_epochs = 2 # 252
	updates = 1
	batch_size = 128
	experiment(series1, series2, series3, series4, updates,look_back,neurons,n_epochs,batch_size)


run()
with open("./loss/cnn-lstm-loss.csv", "w", encoding="utf-8") as fout:
	for losss_data in zip(loss_list):
		fout.writelines(str(losss_data[0])+ "\n")
fig = plt.figure()
plt.plot(loss_list, label='loss', color='blue')
plt.legend()
plt.title('model loss')
plt.savefig('./result/loss-train_no_attention'+file+".png")
plt.show()

