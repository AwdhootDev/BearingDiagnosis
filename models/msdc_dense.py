import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.pad_left = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        x_padded = F.pad(x, (self.pad_left, 0))
        return self.conv(x_padded)

class MSDCBranch(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.01),

            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2*dilation, dilation=dilation),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2*dilation, dilation=dilation)
        )

    def forward(self, x):
        return self.branch(x)

class MSDCModule(nn.Module):
    def __init__(self, in_channels=21):
        super().__init__()
        self.branch1 = MSDCBranch(in_channels, out_channels=21, dilation=1)
        self.branch2 = MSDCBranch(in_channels, out_channels=21, dilation=2)
        self.branch3 = MSDCBranch(in_channels, out_channels=21, dilation=4)

        self.final_bn = nn.BatchNorm1d(63)
        self.final_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x_concat = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)
        return self.final_relu(self.final_bn(x_concat))

class DenseTCNLayer(nn.Module):
    def __init__(self, in_channels, growth_rate=32, base_dilation=1):
        super().__init__()
        inner_channels = 4 * growth_rate 

        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(0.01),
            nn.Conv1d(in_channels, inner_channels, kernel_size=1),
            nn.BatchNorm1d(inner_channels),
            nn.LeakyReLU(0.01)
        )

        self.causal_conv1 = CausalDilatedConv1d(inner_channels, inner_channels, kernel_size=3, dilation=1 * base_dilation)
        self.causal_conv2 = CausalDilatedConv1d(inner_channels, inner_channels, kernel_size=3, dilation=2 * base_dilation)
        self.causal_conv3 = CausalDilatedConv1d(inner_channels, inner_channels, kernel_size=3, dilation=4 * base_dilation)

        self.output_proj = nn.Conv1d(inner_channels, growth_rate, kernel_size=1)

    def forward(self, x):
        out = self.bottleneck(x)
        out = self.causal_conv1(out)
        out = self.causal_conv2(out)
        out = self.causal_conv3(out)
        out = self.output_proj(out)
        return out

class DenseTCNBlock(nn.Module):
    def __init__(self, num_layers=5, in_channels=63, growth_rate=32, base_dilation=1):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for _ in range(num_layers):
            self.layers.append(DenseTCNLayer(current_channels, growth_rate, base_dilation))
            current_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            concat_features = torch.cat(features, dim=1)
            new_features = layer(concat_features)
            features.append(new_features)

        return torch.cat(features, dim=1)

class ParallelDenseTCN(nn.Module):
    def __init__(self, in_channels=63, num_layers=5, growth_rate=32):
        super().__init__()
        self.block1 = DenseTCNBlock(num_layers, in_channels, growth_rate, base_dilation=1)
        self.block2 = DenseTCNBlock(num_layers, in_channels, growth_rate, base_dilation=2)
        self.block3 = DenseTCNBlock(num_layers, in_channels, growth_rate, base_dilation=3)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        return torch.cat([out1, out2, out3], dim=1)

class MSDCDenseTCN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 21, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.msdc = MSDCModule(in_channels=21)

        self.dense_tcn = ParallelDenseTCN(in_channels=63, num_layers=5, growth_rate=32)

        self.classifier_head = nn.Sequential(
            nn.BatchNorm1d(669),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(669, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.msdc(x)
        x = self.dense_tcn(x)
        x = self.classifier_head(x)
        return x
