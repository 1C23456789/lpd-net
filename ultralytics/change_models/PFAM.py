import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# YOLO 基础 Conv（保持不变）
# ---------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ==============================================
#  空间雾掩码生成器（保留 H×W 分辨率）
# ==============================================
class SpatialFogMask(nn.Module):
    """使用轻量卷积生成空间雾浓度掩码，输出形状 (B,1,H,W)"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.mask_conv = nn.Sequential(
            Conv(channels, channels // reduction, 3, act=True),
            Conv(channels // reduction, 1, 3, act=nn.Sigmoid())   # 单通道空间掩码
        )
        # 可学习渐进系数（前期强抑制，后期自动减弱）
        self.prog_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, epoch_ratio=1.0):
        """
        epoch_ratio: 训练进度 [0,1]
        返回空间掩码 (B,1,H,W)，值越大表示该位置雾浓度越高（需抑制）
        """
        # 空间掩码（不经过全局池化，保留空间信息）
        fog_mask = self.mask_conv(x)          # (B,1,H,W)

        # 特征响应分支（保护目标区域：高响应区域掩码值降低）
        # 使用局部最大池化提取显著性，也可用卷积实现
        feat_saliency = torch.max(x, dim=1, keepdim=True)[0]   # (B,1,H,W) 通道最大值作为显著性
        feat_saliency = torch.sigmoid(feat_saliency)           # 归一化
        feat_mask = 1.0 - feat_saliency                         # 高响应区域掩码值小（保护）

        # 融合：雾浓度高且特征不显著的区域才需要强抑制
        fused_mask = fog_mask * feat_mask

        # 渐进强度控制（递减：前期强，后期弱，避免过调制）
        if epoch_ratio < 0.3:
            scale = 1.0
        elif epoch_ratio < 0.7:
            scale = 1.0 - (epoch_ratio - 0.3) / 0.4 * 0.6   # 线性降至0.4
        else:
            scale = 0.4                                       # 后期保持0.4

        alpha = torch.clamp(self.prog_alpha * scale, 0.2, 1.0)
        final_mask = torch.clamp(fused_mask * alpha, 0.0, 0.7)   # 上限0.7避免过度抑制
        return final_mask


# ==============================================
# 空间注意力 FogPSA（保留原始PSA的位置敏感性）
# ==============================================
class FogPSA(nn.Module):
    """空间注意力 + 雾掩码调制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        # 空间注意力生成分支（保持空间结构）
        self.attn_conv = nn.Sequential(
            Conv(channels, channels // reduction, 1),
            Conv(channels // reduction, 1, 1, act=nn.Sigmoid())   # 输出 (B,1,H,W) 空间注意力权重
        )
        # 雾掩码生成器（内部包含渐进强度控制）
        self.mask_gen = SpatialFogMask(channels)
        self._spatial_attn = None
        self._final_mask = None

    def forward(self, x, epoch_ratio=1.0):
        # 1. 生成空间注意力图（原始PSA风格）
        spatial_attn = self.attn_conv(x)          # (B,1,H,W)

        # 2. 生成雾掩码（空间自适应）
        fog_mask = self.mask_gen(x, epoch_ratio)  # (B,1,H,W)，高雾区值大
        self._spatial_attn = spatial_attn.detach()
        self._final_mask = fog_mask.detach()

        # 3. 调制：高雾区降低注意力权重，目标区保持原权重
        modulated_attn = spatial_attn * (1 - fog_mask)   # 雾浓度高 → 注意力低

        # 4. 应用注意力
        return x * modulated_attn


# ==============================================
# PFAMBlock（残差 + FFN）
# ==============================================
class PFAMBlock(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.attn = FogPSA(channels)
        self.ffn = nn.Sequential(
            Conv(channels, channels * 2, 1),
            Conv(channels * 2, channels, 1, act=False)
        )
        self.shortcut = shortcut

    def forward(self, x, epoch_ratio=1.0):
        # 注意力部分（带雾掩码）
        x = x + self.attn(x, epoch_ratio) if self.shortcut else self.attn(x, epoch_ratio)
        # FFN部分
        x = x + self.ffn(x) if self.shortcut else self.ffn(x)
        return x


# ==============================================
#  最终版 PFAM（整合所有改进）
# ==============================================
class PFAM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, shortcut=True):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)                     # 压缩后通道数
        self.cv1 = Conv(c1, 2 * self.c, 1)       # 扩展+拆分
        self.cv2 = Conv(2 * self.c, c1, 1)       # 恢复通道
        # 多个 PFAMBlock 串联（每个内部都包含雾掩码生成）
        self.m = nn.Sequential(*[PFAMBlock(self.c, shortcut) for _ in range(n)])

    def forward(self, x, epoch_ratio=1.0):
        """
        x: 输入特征 (B,C,H,W)
        epoch_ratio: 训练进度（由外部传入，用于渐进强度控制）
        """
        # 1. 扩展并拆分为两路
        a, b = self.cv1(x).split((self.c, self.c), dim=1)   # a:残差分支, b:注意力分支

        # 2. 多个 PFAMBlock 处理 b（每个block内部会自动根据epoch_ratio生成雾掩码）
        b = self.m(b)   # 注意：这里需要将 epoch_ratio 传递给每个 block
        # 由于 self.m 是 Sequential，需要手动修改 forward 或使用自定义循环
        # 重写为循环传递 epoch_ratio（见下方实际 forward 实现）

        # 3. 拼接 + 恢复通道
        out = self.cv2(torch.cat((a, b), dim=1))
        return out

    # 实际使用时需要能够传递 epoch_ratio，因此建议不使用 Sequential，而是手动循环
    # 更灵活的 forward 替代：
    def forward_with_progress(self, x, epoch_ratio=1.0):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        for block in self.m:
            b = block(b, epoch_ratio)   # 每个 block 都接收进度
        return self.cv2(torch.cat((a, b), dim=1))

    # 为了兼容原有调用方式（不传 epoch_ratio 时默认 1.0，即后期模式）
    def forward(self, x, epoch_ratio=1.0):
        return self.forward_with_progress(x, epoch_ratio)


# ---------------------------
# 测试代码
# ---------------------------
if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    model = PFAM(64, 64, n=2)

    # 模拟训练早期（强抑制）
    out_early = model(x, epoch_ratio=0.1)
    # 模拟训练晚期（弱抑制）
    out_late = model(x, epoch_ratio=0.9)

    print(f"早期输出形状: {out_early.shape}, 均值: {out_early.mean().item():.4f}")
    print(f"晚期输出形状: {out_late.shape}, 均值: {out_late.mean().item():.4f}")