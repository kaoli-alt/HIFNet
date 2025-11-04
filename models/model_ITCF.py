import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Starnet
from module import ITCF  # 导入IFB模块


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


class SimpleTemporalFusion(nn.Module):
    """简单的时序融合模块 - 用于低层特征（C1, C2）"""

    def __init__(self, in_channels, out_channels):
        super(SimpleTemporalFusion, self).__init__()
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
    """简单的多尺度解码器 - 使用五层特征"""

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
        # 添加额外的输出层
        self.output1_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output2_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output3_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output4_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        # 最终分类层
        self.final_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4, c5):
        """
        c1, c2, c3, c4, c5: 来自backbone的五层特征
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

        # 上采样到原图尺寸
        output = F.interpolate(p1, scale_factor=2, mode='bilinear', align_corners=False)

        output = torch.sigmoid(self.output1_conv(p1))  # 1/2原始尺寸
        output2 = torch.sigmoid(self.output2_conv(p2))  # 1/4原始尺寸
        output3 = torch.sigmoid(self.output3_conv(p3))  # 1/8原始尺寸
        output4 = torch.sigmoid(self.output4_conv(p4))  # 1/16原始尺寸

        return output, output2, output3, output4


class ChangeDetection(nn.Module):

    def __init__(self, input_nc=3, output_nc=1, window_size=4):
        super(ChangeDetection, self).__init__()

        # 骨干网络
        self.backbone = Starnet.starnet_s3(pretrained=True)

        backbone_channels = [32, 32, 64, 128, 256]
        feature_channels = 64  # 统一的特征通道数

        # 特征适配器：将各层backbone输出统一到固定通道数
        self.feature_adapter1 = SimpleFeatureAdapter(backbone_channels[0], feature_channels)
        self.feature_adapter2 = SimpleFeatureAdapter(backbone_channels[1], feature_channels)
        self.feature_adapter3 = SimpleFeatureAdapter(backbone_channels[2], feature_channels)
        self.feature_adapter4 = SimpleFeatureAdapter(backbone_channels[3], feature_channels)
        self.feature_adapter5 = SimpleFeatureAdapter(backbone_channels[4], feature_channels)

        # 时序融合模块
        # 低语义层（C1, C2）使用简单融合
        self.temporal_fusion1 = SimpleTemporalFusion(feature_channels, feature_channels)
        self.temporal_fusion2 = SimpleTemporalFusion(feature_channels, feature_channels)

        # 高语义层（C3, C4, C5）使用IFB进行互补增强
        self.temporal_fusion3 = ITCF.InterassistedFusionBlock(feature_channels, window_size=window_size)
        self.temporal_fusion4 = ITCF.InterassistedFusionBlock(feature_channels, window_size=window_size)
        self.temporal_fusion5 = ITCF.InterassistedFusionBlock(feature_channels, window_size=window_size)

        # 多尺度解码器
        self.decoder = Decoder(feature_channels, output_nc)


    def forward(self, x1, x2):
        """
        x1, x2: 两个时期的输入图像 [B, 3, H, W]
        返回: 变化检测结果 [B, 1, H, W]
        """
        # 提取特征
        features1 = self.backbone(x1)
        features2 = self.backbone(x2)

        # 获取五层特征
        c1_1, c2_1, c3_1, c4_1, c5_1 = features1
        c1_2, c2_2, c3_2, c4_2, c5_2 = features2

        # 特征适配
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

        # 时序融合
        # 低语义层：简单差值融合
        tf1 = self.temporal_fusion1(f1_1, f1_2)
        tf2 = self.temporal_fusion2(f2_1, f2_2)

        # 高语义层：IFB互补增强融合
        tf3, _ = self.temporal_fusion3(f3_1, f3_2)
        tf4, _ = self.temporal_fusion4(f4_1, f4_2)
        tf5, _ = self.temporal_fusion5(f5_1, f5_2)

        # 获取多尺度输出
        output, output2, output3, output4 = self.decoder(tf1, tf2, tf3, tf4, tf5)

        return output, output2, output3, output4

