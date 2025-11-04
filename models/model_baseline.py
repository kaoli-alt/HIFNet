import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Starnet


class SimpleFeatureAdapter(nn.Module):
    """简单的特征适配器，统一不同层的通道数"""

    def __init__(self, in_channels, out_channels):
        super(SimpleFeatureAdapter, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TemporalFusion(nn.Module):
    """简单的时序融合模块"""

    def __init__(self, in_channels, out_channels):
        super(TemporalFusion, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # 简单的差值融合
        diff = torch.abs(x1 - x2)
        return self.conv(diff)


class Decoder(nn.Module):
    """多尺度解码器 - 返回4个不同尺度的输出用于多尺度监督"""

    def __init__(self, feature_channels, num_classes=1):
        super(Decoder, self).__init__()
        self.feature_channels = feature_channels

        # 各层特征的融合卷积
        self.lateral_conv5 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.lateral_conv4 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.lateral_conv3 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.lateral_conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.lateral_conv1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        # 上采样后的特征精炼
        self.smooth_conv4 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.smooth_conv3 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.smooth_conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)
        self.smooth_conv1 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1)

        # 多尺度输出分类层
        self.output_conv4 = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output_conv3 = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output_conv2 = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output_conv1 = nn.Conv2d(feature_channels, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4, c5):
        """
        c1, c2, c3, c4, c5: 来自backbone的五层特征
        返回4个不同尺度的输出用于多尺度监督
        """
        # 横向连接
        p5 = self.lateral_conv5(c5)
        p4 = self.lateral_conv4(c4) + F.interpolate(p5, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.lateral_conv3(c3) + F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)
        p2 = self.lateral_conv2(c2) + F.interpolate(p3, scale_factor=2, mode='bilinear', align_corners=False)
        p1 = self.lateral_conv1(c1) + F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)

        # 特征精炼
        p4 = self.smooth_conv4(p4)
        p3 = self.smooth_conv3(p3)
        p2 = self.smooth_conv2(p2)
        p1 = self.smooth_conv1(p1)

        # 生成多尺度输出
        # 主输出：上采样到原图尺寸
        output1 = F.interpolate(p1, scale_factor=2, mode='bilinear', align_corners=False)
        output1 = self.output_conv1(output1)

        # 辅助输出：不同尺度用于多尺度监督
        output2 = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        output2 = self.output_conv2(output2)

        output3 = F.interpolate(p3, scale_factor=8, mode='bilinear', align_corners=False)
        output3 = self.output_conv3(output3)

        output4 = F.interpolate(p4, scale_factor=16, mode='bilinear', align_corners=False)
        output4 = self.output_conv4(output4)

        return output1, output2, output3, output4


class ChangeDetection(nn.Module):
    """改进的baseline变化检测模型 - 支持多尺度输出"""

    def __init__(self, input_nc=3, output_nc=1):
        super(ChangeDetection, self).__init__()

        # 骨干网络
        self.backbone = Starnet.starnet_s3(pretrained=True)

        # 原始backbone输出通道数：[32, 32, 64, 128, 256]
        backbone_channels = [32, 32, 64, 128, 256]
        feature_channels = 64  # 统一的特征通道数

        # 特征适配器：将各层backbone输出统一到固定通道数
        self.feature_adapter1 = SimpleFeatureAdapter(backbone_channels[0], feature_channels)
        self.feature_adapter2 = SimpleFeatureAdapter(backbone_channels[1], feature_channels)
        self.feature_adapter3 = SimpleFeatureAdapter(backbone_channels[2], feature_channels)
        self.feature_adapter4 = SimpleFeatureAdapter(backbone_channels[3], feature_channels)
        self.feature_adapter5 = SimpleFeatureAdapter(backbone_channels[4], feature_channels)

        # 时序融合模块 - 为每一层单独设计
        self.temporal_fusion1 = TemporalFusion(feature_channels, feature_channels)
        self.temporal_fusion2 = TemporalFusion(feature_channels, feature_channels)
        self.temporal_fusion3 = TemporalFusion(feature_channels, feature_channels)
        self.temporal_fusion4 = TemporalFusion(feature_channels, feature_channels)
        self.temporal_fusion5 = TemporalFusion(feature_channels, feature_channels)

        # 多尺度解码器
        self.decoder = Decoder(feature_channels, output_nc)

    def forward(self, x1, x2):
        """
        x1, x2: 两个时期的输入图像 [B, 3, H, W]
        返回: 4个不同尺度的变化检测结果用于多尺度监督
        """
        # 提取特征
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)

        # 获取五层特征
        c1_1, c2_1, c3_1, c4_1, c5_1 = features1
        c1_2, c2_2, c3_2, c4_2, c5_2 = features2

        # 特征适配 - 统一通道数
        f1_1 = self.feature_adapter1(c1_1)
        f2_1 = self.feature_adapter2(c2_1)
        f3_1 = self.feature_adapter3(c3_1)
        f4_1 = self.feature_adapter4(c4_1)
        f5_1 = self.feature_adapter5(c5_1)

        f1_2 = self.feature_adapter1(c1_2)
        f2_2 = self.feature_adapter2(c2_2)
        f3_2 = self.feature_adapter3(c3_2)
        f4_2 = self.feature_adapter4(c4_2)
        f5_2 = self.feature_adapter5(c5_2)

        # 时序融合 - 每层单独融合
        tf1 = self.temporal_fusion1(f1_1, f1_2)
        tf2 = self.temporal_fusion2(f2_1, f2_2)
        tf3 = self.temporal_fusion3(f3_1, f3_2)
        tf4 = self.temporal_fusion4(f4_1, f4_2)
        tf5 = self.temporal_fusion5(f5_1, f5_2)

        # 多尺度解码得到4个不同尺度的变化图
        output, output2, output3, output4 = self.decoder(tf1, tf2, tf3, tf4, tf5)

        # 应用sigmoid激活
        output = torch.sigmoid(output)
        output2 = torch.sigmoid(output2)
        output3 = torch.sigmoid(output3)
        output4 = torch.sigmoid(output4)

        return output, output2, output3, output4