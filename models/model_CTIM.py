import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Starnet
from module.CTIM import LightweightCTIM


class EnhancedStarNetBackbone(nn.Module):
    """
    增强的StarNet骨干网络，集成轻量级跨期交互模块

    功能：
    1. 基于StarNet进行多尺度特征提取
    2. 在指定位置插入LightweightCTIM进行跨期特征交互
    3. 支持单时相和双时相前向传播
    """

    def __init__(self, base_starnet, use_ctim=True, ctim_positions=[1, 2, 3]):
        """
        Args:
            base_starnet: 原始的StarNet模型
            use_ctim: 是否使用CTIM模块
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
                    self.ctim_modules[str(pos)] = LightweightCTIM(channels)

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
    """简单的时序融合模块 - 现在用于所有层的特征融合"""

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


class EnhancedTemporalFusion(nn.Module):
    """增强的时序融合模块 - 替代IIFB，用于高语义层"""

    def __init__(self, in_channels, out_channels):
        super(EnhancedTemporalFusion, self).__init__()

        # 差值分支
        self.diff_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 注意力分支
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

        # 最终融合
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # 计算差值特征
        diff = torch.abs(x1 - x2)
        diff_feat = self.diff_conv(diff)

        # 计算注意力权重
        concat_feat = torch.cat([x1, x2], dim=1)
        attention = self.attention_conv(concat_feat)

        # 应用注意力权重
        enhanced_feat = diff_feat * attention

        # 最终特征精炼
        output = self.final_conv(enhanced_feat)

        return output


class Decoder(nn.Module):
    """多尺度解码器 - 使用五层特征进行FPN式解码"""

    def __init__(self, feature_channels, num_classes=1):
        super(Decoder, self).__init__()
        self.feature_channels = feature_channels

        # 各层特征的横向连接
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

        # 多尺度输出层
        self.output1_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output2_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output3_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        self.output4_conv = nn.Conv2d(feature_channels, num_classes, kernel_size=1)

    def forward(self, c1, c2, c3, c4, c5):
        """
        c1, c2, c3, c4, c5: 来自backbone的五层特征
        返回: 四个不同尺度的输出
        """
        # FPN式的自顶向下路径
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
        output1 = torch.sigmoid(self.output1_conv(p1))  # 1/2原始尺寸
        output2 = torch.sigmoid(self.output2_conv(p2))  # 1/4原始尺寸
        output3 = torch.sigmoid(self.output3_conv(p3))  # 1/8原始尺寸
        output4 = torch.sigmoid(self.output4_conv(p4))  # 1/16原始尺寸

        return output1, output2, output3, output4


class CTIMChangeDetection(nn.Module):
    """
    基于CTIM的变化检测模型

    架构特点：
    1. 使用增强的StarNet作为特征提取骨干
    2. 集成CTIM进行跨期特征交互
    3. 使用简单融合和增强融合处理不同语义层次
    4. 采用FPN式多尺度解码器
    """

    def __init__(self, input_nc=3, output_nc=1,
                 use_ctim=True, ctim_positions=[1, 2, 3],
                 use_enhanced_fusion=False):
        """
        Args:
            input_nc: 输入通道数
            output_nc: 输出通道数
            use_ctim: 是否使用CTIM模块
            ctim_positions: CTIM插入位置 (0:stem后, 1-4:各stage后)
            use_enhanced_fusion: 是否对高语义层使用增强融合（替代原IIFB）
        """
        super(CTIMChangeDetection, self).__init__()

        self.use_ctim = use_ctim
        self.use_enhanced_fusion = use_enhanced_fusion

        # 创建增强的StarNet骨干网络
        base_starnet = Starnet.starnet_s3(pretrained=True)
        self.backbone = EnhancedStarNetBackbone(
            base_starnet,
            use_ctim=use_ctim,
            ctim_positions=ctim_positions
        )

        # StarNet各层通道数和统一特征通道数
        backbone_channels = [32, 32, 64, 128, 256]
        feature_channels = 64

        # 特征适配器：将各层backbone输出统一到固定通道数
        self.feature_adapters = nn.ModuleList([
            SimpleFeatureAdapter(backbone_channels[i], feature_channels)
            for i in range(5)
        ])

        # 时序融合模块
        # 低语义层（C1, C2）：使用简单融合
        self.temporal_fusion_low = nn.ModuleList([
            SimpleTemporalFusion(feature_channels, feature_channels),
            SimpleTemporalFusion(feature_channels, feature_channels)
        ])

        # 高语义层（C3, C4, C5）：根据配置使用简单融合或增强融合
        if use_enhanced_fusion:
            self.temporal_fusion_high = nn.ModuleList([
                EnhancedTemporalFusion(feature_channels, feature_channels),
                EnhancedTemporalFusion(feature_channels, feature_channels),
                EnhancedTemporalFusion(feature_channels, feature_channels)
            ])
        else:
            self.temporal_fusion_high = nn.ModuleList([
                SimpleTemporalFusion(feature_channels, feature_channels),
                SimpleTemporalFusion(feature_channels, feature_channels),
                SimpleTemporalFusion(feature_channels, feature_channels)
            ])

        # 多尺度解码器
        self.decoder = Decoder(feature_channels, output_nc)

    def forward(self, x1, x2):
        """
        前向传播

        Args:
            x1, x2: 两个时期的输入图像 [B, 3, H, W]
        Returns:
            tuple: 四个不同尺度的变化检测结果
                - output1: [B, 1, H/2, W/2]
                - output2: [B, 1, H/4, W/4]
                - output3: [B, 1, H/8, W/8]
                - output4: [B, 1, H/16, W/16]
        """
        # 1. 特征提取 + 跨期交互
        if self.use_ctim:
            features1, features2 = self.backbone(x1, x2)
        else:
            features1 = self.backbone(x1)
            features2 = self.backbone(x2)

        # 2. 特征适配
        adapted_features1 = [adapter(feat) for adapter, feat in zip(self.feature_adapters, features1)]
        adapted_features2 = [adapter(feat) for adapter, feat in zip(self.feature_adapters, features2)]

        # 3. 时序融合
        fused_features = []

        # 低语义层：简单差值融合
        for i in range(2):
            fused_feat = self.temporal_fusion_low[i](adapted_features1[i], adapted_features2[i])
            fused_features.append(fused_feat)

        # 高语义层：增强融合或简单融合
        for i in range(3):
            fused_feat = self.temporal_fusion_high[i](adapted_features1[i + 2], adapted_features2[i + 2])
            fused_features.append(fused_feat)

        # 4. 多尺度解码
        output1, output2, output3, output4 = self.decoder(*fused_features)

        return output1, output2, output3, output4

    def get_ctim_attention_maps(self, x1, x2):
        """
        获取CTIM注意力图用于可视化分析

        Returns:
            dict: 各层的注意力图
        """
        if not self.use_ctim:
            return {}

        attention_maps = {}
        features1, features2 = [], []

        # Stem阶段
        f1 = self.backbone.stem(x1)
        f2 = self.backbone.stem(x2)

        if '0' in self.backbone.ctim_modules:
            concat_feat = torch.cat([f1, f2], dim=1)
            attention_map = self.backbone.ctim_modules['0'].cross_attn(concat_feat)
            attention_maps['stem'] = attention_map

        features1.append(f1)
        features2.append(f2)

        # 各个Stage阶段
        for i, stage in enumerate(self.backbone.stages):
            f1 = stage(f1)
            f2 = stage(f2)

            stage_pos = str(i + 1)
            if stage_pos in self.backbone.ctim_modules:
                concat_feat = torch.cat([f1, f2], dim=1)
                attention_map = self.backbone.ctim_modules[stage_pos].cross_attn(concat_feat)
                attention_maps[f'stage_{i + 1}'] = attention_map

        return attention_maps

