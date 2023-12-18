import torch
from torchsummary import summary
 # Make sure to replace 'your_script_file' with the actual filename
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.lstm1 = nn.LSTM(input_size=99, hidden_size=128, num_layers=1, batch_first=True) #input은  45 * 3(x, y z)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm4 = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
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
# Load the model
model = Model1()
model.load_state_dict(torch.load(r'C:\Users\USER\Desktop\19rne\2023-RnE-main\save_by_loss\model_len30_by_loss.pth'))

# Print the model summary
summary(model, input_size=(32, 75))  # Replace batch_size and input_size with appropriate values

#%%
