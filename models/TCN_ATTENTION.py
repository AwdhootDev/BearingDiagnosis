import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size): 
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, padding):
        super().__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding = padding, dilation = dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding = padding, dilation = dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.conv2,
            self.chomp2,
            self.relu2
        )

        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNAttention(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.tcn = nn.Sequential(
            TemporalBlock(1, 64, kernel_size=3, dilation=1, padding=2),
            TemporalBlock(64, 128, kernel_size=3, dilation=2, padding=4),
            TemporalBlock(128, 256, kernel_size=3, dilation=4, padding=8)
        )

        self.attention = nn.Linear(256, 1)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights*x, dim=1)
        return self.fc(context)

def load_model_ta(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TCNAttention().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))

    model.eval()

    return model, device
