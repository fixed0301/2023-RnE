import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataset): #모든 행동을 통합한 df가 들어가야함
        self.x = []
        self.y = []
        for dic in dataset:
            self.y.append(dic['key']) #key 값에는 actions 들어감
            self.x.append(dic['value']) #action마다의 data 들어감

    def __getitem__(self, index): #index는 행동의 index
        data = self.x[index] # x에는 꺼내 쓸 (행동마다 45개 묶음프레임)의 데이터
        label = self.y[index]
        return torch.Tensor(np.array(data)), torch.tensor(np.array(int(label)))

    def __len__(self):
        return len(self.x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm1 = nn.LSTM(input_size=99, hidden_size=128, num_layers=1, batch_first=True) #input은  45 * 3(x, y z)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0, 1)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0, 1)
        self.lstm7 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 5) #분류할 클래스 5가지

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = self.dropout1(x)
        x, _ = self.lstm4(x)
        x, _ = self.lstm5(x)
        x, _ = self.lstm6(x)
        x = self.dropout2(x)
        x, _ = self.lstm7(x)
        x = self.fc(x[:, -1, :])
        return x

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model