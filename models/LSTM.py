import torch
import torch.nn as nn

class BearingLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size1=128, hidden_size2=64, num_classes=4):
        super(BearingLSTM, self).__init__()
    
        self.lstm1 = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size1,
            num_layers = 1,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size = hidden_size1,
            hidden_size = hidden_size2,
            num_layers = 1,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)

        out = out[:, -1, :]

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

def load_model_lstm(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BearingLSTM().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))

    model.eval()

    return model, device