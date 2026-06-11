import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

Conv2d = nn.Conv2d


# Layer Norm (保持原有代码不变)
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# 泰勒注意力模块 (Taylor Attention)
class TaylorAttention(nn.Module):
    def __init__(self, dim, num_heads=8, taylor_terms=3, bias=False):
        super(TaylorAttention, self).__init__()
        self.num_heads = num_heads
        self.taylor_terms = taylor_terms
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 泰勒展开的系数学习
        self.taylor_coeffs = nn.Parameter(torch.ones(taylor_terms))

        # 投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)

        # 相对位置编码
        self.rel_pos_embedding = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim))

    def taylor_softmax(self, x, dim=-1):
        """泰勒展开近似softmax"""
        # 泰勒展开: exp(x) ≈ 1 + x + x^2/2! + x^3/3! + ...
        result = torch.ones_like(x)
        x_power = x.clone()
        factorial = 1.0

        for i in range(1, self.taylor_terms + 1):
            factorial *= i
            term = x_power * self.taylor_coeffs[i - 1] / factorial
            result = result + term
            x_power = x_power * x

        return result / (result.sum(dim=dim, keepdim=True) + 1e-6)

    def forward(self, x):
        B, N, C = x.shape

        # 生成QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加相对位置编码
        attn = attn + self.rel_pos_embedding

        # 使用泰勒softmax
        attn = self.taylor_softmax(attn, dim=-1)

        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


# 增强的Dual-scale Gated Feed-Forward Network with Taylor Expansion
class TaylorEnhancedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.5, bias=False, taylor_terms=2):
        super(TaylorEnhancedFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.taylor_terms = taylor_terms

        self.project_in = Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 多尺度卷积
        self.dwconv_3 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3,
                               stride=1, padding=1, groups=hidden_features // 4, bias=bias)
        self.dwconv_5 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5,
                               stride=1, padding=2, groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3,
                                     stride=1, padding=2, dilation=2, groups=hidden_features // 4, bias=bias)

        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        # 泰勒增强的激活函数
        self.taylor_activation = TaylorActivation(taylor_terms=taylor_terms)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        # 分割特征并应用不同尺度的卷积
        x1, x2, x3, x4 = x.chunk(4, dim=1)

        x1 = self.dwconv_3(x1)
        x2 = self.dwconv_5(x2)
        x3 = self.dwconv_dilated(x3)

        # 泰勒增强的交互
        x_combined = self.taylor_activation(x1) * x2 + self.taylor_activation(x2) * x3
        x_combined = torch.cat([x_combined, x4], dim=1)

        x = self.p_unshuffle(x_combined)
        x = self.project_out(x)

        return x


class TaylorActivation(nn.Module):
    """泰勒展开增强的激活函数"""

    def __init__(self, taylor_terms=3):
        super(TaylorActivation, self).__init__()
        self.taylor_terms = taylor_terms
        self.coeffs = nn.Parameter(torch.ones(taylor_terms))

    def forward(self, x):
        result = x.clone()
        x_power = x.clone()

        for i in range(2, self.taylor_terms + 1):
            x_power = x_power * x
            result = result + self.coeffs[i - 1] * x_power / math.factorial(i)

        return F.mish(result)


# 创新的泰勒直方图自注意力 (Taylor Histogram Self-Attention)
class TaylorHistogramAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, taylor_terms=3, ifBox=True):
        super(TaylorHistogramAttention, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.taylor_terms = taylor_terms

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.taylor_coeffs = nn.Parameter(torch.ones(taylor_terms))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)

        # 泰勒注意力分支
        self.taylor_attn = TaylorAttention(dim // 2, num_heads // 2, taylor_terms)

        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def taylor_softmax_1(self, x, dim=-1):
        """泰勒展开的softmax近似"""
        result = torch.ones_like(x)
        x_power = x.clone()
        factorial = 1.0

        for i in range(1, self.taylor_terms + 1):
            factorial *= i
            term = x_power * self.taylor_coeffs[i - 1] / factorial
            result = result + term
            x_power = x_power * x

        return result / (result.sum(dim, keepdim=True) + 1e-6)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.taylor_softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        # 直方图排序处理
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = x.clone()
        x[:, :c // 2] = x_sort

        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        # 泰勒注意力分支
        v_flat = v.view(b, c, -1).permute(0, 2, 1)
        taylor_out = self.taylor_attn(v_flat[:, :, :c // 2])
        taylor_out = taylor_out.permute(0, 2, 1).view(b, c // 2, h, w)

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)

        # 融合泰勒注意力输出
        out = out1 * out2
        out[:, :c // 2] = out[:, :c // 2] + taylor_out

        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace

        return out


# 创新的泰勒直方图Transformer块 (Taylor Histogram Transformer Block)
class TaylorHistogramTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.5, bias=False,
                 LayerNorm_type='WithBias', taylor_terms=3):
        super(TaylorHistogramTransformerBlock, self).__init__()

        # 泰勒增强的注意力
        self.attn_g = TaylorHistogramAttention(dim, num_heads, bias, taylor_terms, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)

        # 泰勒增强的FFN
        self.ffn = TaylorEnhancedFeedForward(dim, ffn_expansion_factor, bias, taylor_terms)
        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

        # 残差连接的泰勒加权
        self.residual_weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        # 泰勒加权的残差连接
        residual1 = x
        x = x + self.residual_weights[0] * self.attn_g(self.norm_g(x))
        x_out = x + self.residual_weights[1] * self.ffn(self.norm_ff1(x))

        return x_out


# Conv类 (保持原有代码不变)
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

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# 创新的C2THTB模块 (基于C2PSA架构但使用THTB)
class C2THTB(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, taylor_terms=3):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 使用泰勒直方图Transformer块
        self.m = nn.Sequential(*(
            TaylorHistogramTransformerBlock(
                self.c,
                num_heads=max(1, self.c // 64),
                taylor_terms=taylor_terms
            ) for _ in range(n)
        ))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# 混合版本：结合泰勒注意力和原有DHSA
class C2Hybrid_THTB_DHSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, taylor_ratio=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.taylor_ratio = taylor_ratio
        self.taylor_channels = int(self.c * taylor_ratio)
        self.dhsa_channels = self.c - self.taylor_channels

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 泰勒分支
        self.taylor_branch = nn.Sequential(*(
            TaylorHistogramTransformerBlock(
                self.taylor_channels,
                num_heads=max(1, self.taylor_channels // 64)
            ) for _ in range(n)
        ))

        # DHSA分支 (保持原有)
        self.dhsa_branch = nn.Sequential(*(
            TransformerBlock(self.dhsa_channels) for _ in range(n)
        ))

        # 特征融合
        self.fusion = Conv(self.c, self.c, 3, 1, 1)

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 分割特征到两个分支
        b_taylor, b_dhsa = b.split([self.taylor_channels, self.dhsa_channels], dim=1)

        b_taylor = self.taylor_branch(b_taylor)
        b_dhsa = self.dhsa_branch(b_dhsa)

        # 融合两个分支的输出
        b_fused = torch.cat([b_taylor, b_dhsa], dim=1)
        b_fused = self.fusion(b_fused)

        return self.cv2(torch.cat((a, b_fused), 1))


if __name__ == '__main__':
    # 测试新的泰勒直方图Transformer块
    THTB = TaylorHistogramTransformerBlock(256, taylor_terms=3)

    # 创建输入张量
    batch_size = 8
    input_tensor = torch.randn(batch_size, 256, 64, 64)

    # 运行模型并打印输入和输出的形状
    output_tensor = THTB(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

    # 测试C2THTB模块
    c2thtb = C2THTB(256, 256, n=2, taylor_terms=3)
    output_c2thtb = c2thtb(input_tensor)
    print("C2THTB Output shape:", output_c2thtb.shape)

    # 测试混合版本
    hybrid = C2Hybrid_THTB_DHSA(256, 256, n=2, taylor_ratio=0.5)
    output_hybrid = hybrid(input_tensor)
    print("Hybrid Output shape:", output_hybrid.shape)