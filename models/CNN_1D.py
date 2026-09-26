
import torch.nn as nn

class FaultDetectionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(FaultDetectionCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=16),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=8),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=4),
            nn.Tanh(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),  # Set to 4 classes
        )

    def forward(self, x):
        x = self.network(x)
        x = self.classifier(x)
        return x