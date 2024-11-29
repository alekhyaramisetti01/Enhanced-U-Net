import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.self_attention = SelfAttention(out_channels)
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
        out += self.shortcut(x)
        return F.relu(out)

class MultiAttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiAttentionUNet, self).__init__()
        self.enc1 = ResidualAttentionBlock(in_channels, 64)
        self.enc2 = ResidualAttentionBlock(64, 128)
        self.enc3 = ResidualAttentionBlock(128, 256)
        self.enc4 = ResidualAttentionBlock(256, 512)
        self.bottom = ResidualAttentionBlock(512, 1024)

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.dec4 = ResidualAttentionBlock(1024, 512)
        self.dec3 = ResidualAttentionBlock(512, 256)
        self.dec2 = ResidualAttentionBlock(256, 128)
        self.dec1 = ResidualAttentionBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        bottom = self.bottom(F.max_pool2d(enc4, 2))

        up4 = self.up1(bottom)
        dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
        up3 = self.up2(dec4)
        dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
        up2 = self.up3(dec3)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up4(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        return self.final(dec1)

if __name__ == "__main__":
    model = MultiAttentionUNet(in_channels=3, out_channels=1)
    test_input = torch.randn(1, 3, 128, 128)  # Smaller input size
    test_output = model(test_input)
    print("Output shape:", test_output.shape)
