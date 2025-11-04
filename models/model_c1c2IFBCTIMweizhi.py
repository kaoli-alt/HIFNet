import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Starnet
from module import ITCF  # 导入IFB模块


# ======================== CTIM模块 ========================

class LightweightCTIM(nn.Module):
    """轻量级跨期交互模块"""

    def __init__(self, channels, reduction_ratio=4):
        super(LightweightCTIM, self).__init__()
        reduced_channels = max(channels // reduction_ratio, 8)

        # 简化的交叉注意力
        self.cross_attn = nn.Sequential(
            nn.Conv2d(channels * 3, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )

        # 差分增强
        self.diff_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f1, f2):
        # 计算跨期注意力权重
        # concat_feat = torch.cat([f1, f2], dim=1)
        # attention_weight = self.cross_attn(concat_feat)

        # 差分特征增强
        diff_feat = torch.abs(f1 - f2)
        concat_feat = torch.cat([f1, f2, diff_feat], dim=1)
        attention_weight = self.cross_attn(concat_feat)
        enhanced_diff = self.diff_conv(diff_feat * attention_weight)

        # 残差连接
        f1_out = f1 + self.gamma * enhanced_diff
        f2_out = f2 + self.gamma * enhanced_diff

        return f1_out, f2_out


# ======================== 增强的骨干网络包装器 ========================
class EnhancedStarNetBackbone(nn.Module):
    """增强的StarNet骨干网络包装器，集成跨期交互模块"""

    def __init__(self, base_starnet, use_ctim=True, ctim_positions=[3, 4]):
        """
        Args:
            base_starnet: 原始的StarNet模型
            use_ctim: 是否使用CTIM
            ctim_positions: CTIM插入位置 (0:stem后, 1-4:各stage后)
        """
        super(EnhancedStarNetBackbone, self).__init__()
        self.use_ctim = use_ctim
        self.ctim_positions = ctim_positions

        # 复制原始StarNet的组件
        self.stem = base_starnet.stem
        self.stages = base_starnet.stages

        # 获取各阶段的通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            stage_channels = self._get_stage_channels(dummy_input)

        # 为指定位置添加CTIM模块
        if self.use_ctim:
            self.ctim_modules = nn.ModuleDict()
            for pos in ctim_positions:
                if 0 <= pos < len(stage_channels):
                    channels = stage_channels[pos]
                    ctim = LightweightCTIM(channels)
                    self.ctim_modules[str(pos)] = ctim

    def _get_stage_channels(self, x):
        """获取各阶段输出的通道数"""
        channels = []
        x = self.stem(x)
        channels.append(x.shape[1])

        for stage in self.stages:
            x = stage(x)
            channels.append(x.shape[1])

        return channels

    def forward_single(self, x):
        """单时相前向传播（兼容原始接口）"""
        features = []
        x = self.stem(x)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features

    def forward_dual(self, x1, x2):
        """双时相前向传播（CTIM模式）"""
        if not self.use_ctim:
            # 如果不使用CTIM，退化为独立前向传播
            return self.forward_single(x1), self.forward_single(x2)

        features1, features2 = [], []

        # Stem阶段
        f1 = self.stem(x1)
        f2 = self.stem(x2)

        # 应用CTIM（位置0：stem后）
        if '0' in self.ctim_modules:
            f1, f2 = self.ctim_modules['0'](f1, f2)

        features1.append(f1)
        features2.append(f2)

        # 各个Stage阶段
        for i, stage in enumerate(self.stages):
            # 特征提取
            f1 = stage(f1)
            f2 = stage(f2)

            # 应用CTIM（位置i+1：第i个stage后）
            stage_pos = str(i + 1)
            if stage_pos in self.ctim_modules:
                f1, f2 = self.ctim_modules[stage_pos](f1, f2)

            features1.append(f1)
            features2.append(f2)

        return features1, features2

    def forward(self, x1, x2=None):
        """
        灵活的前向传播接口
        - 如果只传入x1，执行单时相前向传播
        - 如果传入x1和x2，执行双时相前向传播
        """
        if x2 is None:
            return self.forward_single(x1)
        else:
            return self.forward_dual(x1, x2)


# ======================== 保持原有组件不变 ========================
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

        # 多尺度输出
        output = torch.sigmoid(self.output1_conv(p1))  # 1/2原始尺寸
        output2 = torch.sigmoid(self.output2_conv(p2))  # 1/4原始尺寸
        output3 = torch.sigmoid(self.output3_conv(p3))  # 1/8原始尺寸
        output4 = torch.sigmoid(self.output4_conv(p4))  # 1/16原始尺寸

        return output, output2, output3, output4


# ======================== 增强的变化检测模型 ========================
class ChangeDetection(nn.Module):
    """原始的变化检测模型（保持兼容性）"""

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
        self.temporal_fusion3 = IIFB.ImprovedInterassistedFusionBlock(feature_channels, window_size=window_size)
        self.temporal_fusion4 = IIFB.ImprovedInterassistedFusionBlock(feature_channels, window_size=window_size)
        self.temporal_fusion5 = IIFB.ImprovedInterassistedFusionBlock(feature_channels, window_size=window_size)

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


class CTIMChangeDetection(nn.Module):
    """集成CTIM的增强变化检测模型"""

    def __init__(self, input_nc=3, output_nc=1, window_size=4,
                 use_ctim=True, ctim_positions=[ 3, 4]):
        """
        Args:
            input_nc: 输入通道数
            output_nc: 输出通道数
            window_size: IFB窗口大小
            use_ctim: 是否使用CTIM
            ctim_positions: CTIM插入位置 (0:stem后, 1-4:各stage后)
        """
        super().__init__()
        self.use_ctim = use_ctim

        # 使用增强的骨干网络
        base_starnet = Starnet.starnet_s3(pretrained=True)
        self.backbone = EnhancedStarNetBackbone(
            base_starnet,
            use_ctim=use_ctim,
            ctim_positions=ctim_positions,
        )

        backbone_channels = [32, 32, 64, 128, 256]
        feature_channels = 64  # 统一的特征通道数

        # 特征适配器：将各层backbone输出统一到固定通道数
        self.feature_adapter1 = SimpleFeatureAdapter(backbone_channels[0], feature_channels)
        self.feature_adapter2 = SimpleFeatureAdapter(backbone_channels[1], feature_channels)
        self.feature_adapter3 = SimpleFeatureAdapter(backbone_channels[2], feature_channels)
        self.feature_adapter4 = SimpleFeatureAdapter(backbone_channels[3], feature_channels)
        self.feature_adapter5 = SimpleFeatureAdapter(backbone_channels[4], feature_channels)

        # 时序融合模块
        # 低语义层（C1, C2）使用简单融合（因为已经过CTIM交互）
        self.temporal_fusion1 = SimpleTemporalFusion(feature_channels, feature_channels)
        self.temporal_fusion2 = SimpleTemporalFusion(feature_channels, feature_channels)

        # 高语义层（C3, C4, C5）使用IFB进行互补增强
        self.temporal_fusion3 = IIFB.ImprovedInterassistedFusionBlock(feature_channels, window_size=window_size)
        self.temporal_fusion4 = IIFB.ImprovedInterassistedFusionBlock(feature_channels, window_size=window_size)
        self.temporal_fusion5 = IIFB.ImprovedInterassistedFusionBlock(feature_channels, window_size=window_size)

        # 多尺度解码器
        self.decoder = Decoder(feature_channels, output_nc)

    def forward(self, x1, x2):
        """
        x1, x2: 两个时期的输入图像 [B, 3, H, W]
        返回: 变化检测结果 [B, 1, H, W]
        """
        # 通过增强骨干网络提取特征（已集成跨期交互）
        if self.use_ctim:
            features1, features2 = self.backbone(x1, x2)
        else:
            # 如果不使用CTIM，使用原始方式
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
        # 低语义层：简单差值融合（已经过CTIM交互增强）
        tf1 = self.temporal_fusion1(f1_1, f1_2)
        tf2 = self.temporal_fusion2(f2_1, f2_2)

        # 高语义层：IFB互补增强融合
        tf3, _ = self.temporal_fusion3(f3_1, f3_2)
        tf4, _ = self.temporal_fusion4(f4_1, f4_2)
        tf5, _ = self.temporal_fusion5(f5_1, f5_2)

        # 获取多尺度输出
        output, output2, output3, output4 = self.decoder(tf1, tf2, tf3, tf4, tf5)

        return output, output2, output3, output4


