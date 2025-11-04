import torch
import torch.nn as nn
import torch.nn.functional as F
from . import Starnet


class ChangeDetectionBasic(nn.Module):
    """最基础的变化检测模型 - 仅使用StarNet特征提取和简单融合"""

    def __init__(self, input_nc=3, output_nc=1):
        super(ChangeDetectionBasic, self).__init__()

        # 1. 骨干网络 - StarNet特征提取
        self.backbone = Starnet.starnet_s3(pretrained=True)

        # 2. 简单融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),  # 降维到64通道
            nn.ReLU(inplace=True)
        )

        # 3. 预测头
        self.head = nn.Conv2d(64, output_nc, kernel_size=1)

    def forward(self, x1, x2):
        """
        x1, x2: 两个时期的输入图像 [B, 3, H, W]
        返回: 单张变化检测图
        """
        # 只提取最后一层特征（深层语义特征）
        _, _, _, _, c5_1 = self.backbone(x1)  # [B, 256, H/32, W/32]
        _, _, _, _, c5_2 = self.backbone(x2)  # [B, 256, H/32, W/32]

        # 简单特征融合 - 绝对差值
        diff = torch.abs(c5_1 - c5_2)

        # 融合降维
        fused = self.fusion(diff)

        # 预测头
        out = self.head(fused)

        # 上采样到原图尺寸
        out = F.interpolate(out, scale_factor=32, mode='bilinear', align_corners=False)

        # 应用sigmoid激活
        return torch.sigmoid(out)