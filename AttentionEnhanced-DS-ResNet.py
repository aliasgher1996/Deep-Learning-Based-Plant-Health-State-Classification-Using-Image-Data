import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.ds_conv1 = DepthwiseSeparableConv(channels, channels, dilation)
        self.ds_conv2 = DepthwiseSeparableConv(channels, channels, dilation)
        self.add = nn.Sequential()  # Identity mapping

    def forward(self, x):
        residual = x
        out = self.ds_conv1(x)
        out = self.ds_conv2(out)
        out += residual
        return out

class AttentionEnhancedDSResNet(nn.Module):
    def __init__(self, num_classes=5):
        super(AttentionEnhancedDSResNet, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(64)
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, dilation=2**(i//3)) for i in range(7)]
        )

        self.additional_ds_conv = DepthwiseSeparableConv(64, 64, dilation=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.se(x)
        x = self.residual_blocks(x)
        x = self.additional_ds_conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Example usage
model = AttentionEnhancedDSResNet(num_classes=3)