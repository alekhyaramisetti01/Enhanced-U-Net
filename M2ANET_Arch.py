import torch
import torch.nn as nn
import torch.nn.functional as F

# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        return self.gamma * out + x


# Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        return x * avg_out


# Spatial Attention Block
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


# Residual Block with Attention
class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.self_attention = SelfAttention(out_channels)
        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.self_attention(out)
        out = self.channel_attention(out)
        out = self.spatial_attention(out)
        out += self.shortcut(x)
        return F.relu(out)


# Multi-Attention U-Net Architecture
class MultiAttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiAttentionUNet, self).__init__()
        self.enc1 = ResidualAttentionBlock(in_channels, 64)
        self.enc2 = ResidualAttentionBlock(64, 128)
        self.enc3 = ResidualAttentionBlock(128, 256)
        self.enc4 = ResidualAttentionBlock(256, 512)
        self.bottom = ResidualAttentionBlock(512, 1024)
        
        # Decoder layers
        self.dec4 = ResidualAttentionBlock(1024 + 512, 512)
        self.dec3 = ResidualAttentionBlock(512 + 256, 256)
        self.dec2 = ResidualAttentionBlock(256 + 128, 128)
        self.dec1 = ResidualAttentionBlock(128 + 64, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Pooling and Upsampling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_layers = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottom
        bottom = self.bottom(self.pool(enc4))

        # Decoder
        up4 = self.up(bottom)  # Upsample first
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))  # Concatenate with corresponding encoder output
        
        up3 = self.up_layers[0](dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        
        up2 = self.up_layers[1](dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.up_layers[2](dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Final output
        return self.final(dec1)
