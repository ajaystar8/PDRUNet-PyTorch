import torch.nn as nn
from config import *


class StandardBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=dilation)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", dilation=dilation)
        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        return self.act(self.bnorm2(self.second(self.act(self.bnorm1(self.first(x))))))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()

        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=dilation)
        self.bnorm1 = nn.BatchNorm2d(out_channels)

        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", dilation=dilation)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()
        self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.bnorm2(self.second(self.act(self.bnorm1(self.first(x)))))
        return self.act(x + shortcut)


class DropoutBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.drop = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor):
        return self.drop(x)


class PDRUNet(nn.Module):
    def __init__(self, in_channels, num_filters, out_channels):
        super().__init__()

        self.std_block = StandardBlock(in_channels, num_filters, 1)

        # encoder block
        self.e1 = ResidualBlock(num_filters, num_filters, 2)
        self.e2 = ResidualBlock(num_filters, num_filters, 4)
        self.e3 = ResidualBlock(num_filters, num_filters, 8)
        self.e4 = ResidualBlock(num_filters, num_filters, 16)
        self.e5 = ResidualBlock(num_filters, num_filters, 32)

        # bottleneck 
        self.bottleneck = nn.Sequential(
            ResidualBlock(num_filters, num_filters, 32),
            DropoutBlock(),
            ResidualBlock(num_filters, num_filters, 32),
            DropoutBlock(),
            ResidualBlock(num_filters, num_filters, 32),
        )

        # decoder block
        self.d1 = ResidualBlock(num_filters, num_filters, 16)
        self.d2 = ResidualBlock(num_filters, num_filters, 8)
        self.d3 = ResidualBlock(num_filters, num_filters, 4)
        self.d4 = ResidualBlock(num_filters, num_filters, 2)
        self.d5 = ResidualBlock(num_filters, num_filters, 1)

        # final conv layer
        self.final_conv = nn.Conv2d(num_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        pass_through = []

        # encoder
        x = self.std_block(x)
        pass_through.append(x)

        x = self.e1(x)
        pass_through.append(x)
        x = x + pass_through[0]

        x = self.e2(x)
        pass_through.append(x)

        x = self.e3(x)
        pass_through.append(x)
        x = x + pass_through[2]

        x = self.e4(x)
        pass_through.append(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        skip = [x]

        x = x + pass_through.pop()
        x = self.d1(x)
        x = x + skip.pop()

        x = x + pass_through.pop()
        x = self.d2(x)
        skip.append(x)

        x = x + pass_through.pop()
        x = self.d3(x)
        x = x + skip.pop()

        x = x + pass_through.pop()
        x = self.d4(x)

        x = x + pass_through.pop()
        x = self.d5(x)

        return self.final_conv(x)
