import torch
import torch.nn as nn

class LSTMModel(nn.Module):

    def __init__(self, window=660, dim=4, lstm_units=660, num_layers=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(input_size=16,hidden_size=lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x

class CNNLSTMModel(nn.Module):

    def __init__(self, window=660, dim=4, lstm_units=660, num_layers=2):
        super(CNNLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(input_size=16,hidden_size=lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        # print('ssssss',x.shape)
        return x


class CNNLSTMModel_ECA(nn.Module):

    def __init__(self, window=660, dim=4, lstm_units=660, num_layers=2):
        super(CNNLSTMModel_ECA, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(input_size=16,hidden_size=lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.attn = nn.Linear(lstm_units * 2, lstm_units * 2)
        self.act3 = nn.Sigmoid()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)
        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        attn = self.attn(x)  # bs, 2*lstm_units
        attn = self.act3(attn)
        x = x * attn
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_SE(nn.Module):

    def __init__(self, window=660, dim=4, lstm_units=660, num_layers=2):
        super(CNNLSTMModel_SE, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_units, batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.se_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        # print('1', x.shape)
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        # print('2', x.shape)
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        # print('3', x.shape)
        x = self.act1(x)
        # print('4', x.shape)
        # se
        avg = x.mean(dim=1)  # bs, window
        # print('5', avg.shape)
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        # print('6', se_attn.shape)
        x = torch.einsum("bnd,bd->bnd", x, se_attn)
        # print('7', x.shape)

        x = self.maxPool(x)  # bs, lstm_units, 1
        # print('8', x.shape)
        x = self.drop(x)
        # print('9', x.shape)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        # print('10', x.shape)
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        # print('11', x.shape)
        x = self.act2(x)
        # print('12', x.shape)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        # print('13', x.shape)
        x = self.cls(x)
        # print('14', x.shape)
        x = self.act4(x)
        # print('15', x.shape)
        return x


class CNNLSTMModel_CBAM(nn.Module):

    def __init__(self, window=660, dim=4, lstm_units=660, num_layers=2):
        super(CNNLSTMModel_CBAM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_units,  batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.se_fc = nn.Linear(window, window)
        self.hw_fc = nn.Linear(16, 16)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # chanal
        avg = x.mean(dim=1)  # bs, window
        # print(avg.shape)
        se_attn = self.se_fc(avg).softmax(dim=-1)  # bs, window
        x = torch.einsum("bnd,bd->bnd", x, se_attn)

        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        # print('avg',avg.shape)
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        # print('hw_attn',hw_attn.shape)
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)

        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x


class CNNLSTMModel_HW(nn.Module):

    def __init__(self, window=660, dim=4, lstm_units=16, num_layers=2):
        super(CNNLSTMModel_HW, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=1)
        self.act1 = nn.Sigmoid()
        self.maxPool = nn.MaxPool1d(kernel_size=window)
        self.drop = nn.Dropout(p=0.01)
        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_units,  batch_first=True, num_layers=1, bidirectional=True)
        self.act2 = nn.Tanh()
        self.cls = nn.Linear(lstm_units * 2, 1)
        self.act4 = nn.Tanh()

        self.hw_fc = nn.Linear(lstm_units, lstm_units)

    def forward(self, x):
        x = x.transpose(-1, -2)  # tf和torch纬度有点不一样
        x = self.conv1d(x)  # in： bs, dim, window out: bs, lstm_units, window
        x = self.act1(x)

        # wh
        avg = x.mean(dim=2)  # bs, lstm_units
        hw_attn = self.hw_fc(avg).softmax(dim=-1)  # bs, lstm_units
        x = torch.einsum("bnd,bn->bnd", x, hw_attn)

        x = self.maxPool(x)  # bs, lstm_units, 1
        x = self.drop(x)
        x = x.transpose(-1, -2)  # bs, 1, lstm_units
        x, (_, _) = self.lstm(x)  # bs, 1, 2*lstm_units
        x = self.act2(x)
        x = x.squeeze(dim=1)  # bs, 2*lstm_units
        x = self.cls(x)
        x = self.act4(x)
        return x
