import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False
    print("[Info] Thư viện segmentation_models_pytorch chưa cài đặt. Hệ thống sẽ tự động sử dụng Fallback Models bằng PyTorch thuần.")

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_l = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_l(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class PureUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        x4 = self.enc4(p3)
        p4 = self.pool4(x4)
        
        b = self.bottleneck(p4)
        
        d4 = self.up4(b)
        if d4.shape[2:] != x4.shape[2:]:
            d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=True)
        d4 = torch.cat((d4, x4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape[2:] != x3.shape[2:]:
            d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[2:] != x1.shape[2:]:
            d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.cat((d1, x1), dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final_conv(d1)
        return logits

class PureAttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = ConvBlock(512, 1024)
        
        self.ag4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.ag3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.ag2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.ag1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        p1 = self.pool1(x1)
        x2 = self.enc2(p1)
        p2 = self.pool2(x2)
        x3 = self.enc3(p2)
        p3 = self.pool3(x3)
        x4 = self.enc4(p3)
        p4 = self.pool4(x4)
        
        b = self.bottleneck(p4)
        
        d4 = self.up4(b)
        if d4.shape[2:] != x4.shape[2:]:
            d4 = F.interpolate(d4, size=x4.shape[2:], mode='bilinear', align_corners=True)
        x4_attn = self.ag4(g=d4, x=x4)
        d4 = torch.cat((d4, x4_attn), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape[2:] != x3.shape[2:]:
            d3 = F.interpolate(d3, size=x3.shape[2:], mode='bilinear', align_corners=True)
        x3_attn = self.ag3(g=d3, x=x3)
        d3 = torch.cat((d3, x3_attn), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x2_attn = self.ag2(g=d2, x=x2)
        d2 = torch.cat((d2, x2_attn), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[2:] != x1.shape[2:]:
            d1 = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x1_attn = self.ag1(g=d1, x=x1)
        d1 = torch.cat((d1, x1_attn), dim=1)
        d1 = self.dec1(d1)
        
        logits = self.final_conv(d1)
        return logits

def get_segmentation_model(
    architecture_name: str = 'attention_unet',
    backbone_name: str = 'resnet50',
    encoder_weights: str = 'imagenet',
    in_channels: int = 1,
    classes: int = 1
) -> nn.Module:
    if HAS_SMP:
        if architecture_name == 'unet':
            return smp.Unet(
                encoder_name=backbone_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes
            )
        elif architecture_name == 'attention_unet':
            return smp.Unet(
                encoder_name=backbone_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                decoder_attention_type='scse'
            )
        elif architecture_name == 'unetplusplus':
            return smp.UnetPlusPlus(
                encoder_name=backbone_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes
            )
        else:
            raise ValueError(f"Không hỗ trợ kiến trúc từ SMP: {architecture_name}")
    else:
        if architecture_name == 'unet':
            return PureUNet(in_channels=in_channels, out_channels=classes)
        elif architecture_name == 'attention_unet':
            return PureAttentionUNet(in_channels=in_channels, out_channels=classes)
        else:
            print(f"[Warning] Không có SMP, tự động chuyển kiến trúc {architecture_name} về Baseline U-Net.")
            return PureUNet(in_channels=in_channels, out_channels=classes)
