import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

Conv2d = nn.Conv2d


## Layer Norm
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
        self.normalized_shape = normalized_shape

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


##########################################################################
## Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_5 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2,
                               groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2,
                                        groups=hidden_features // 4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x


##########################################################################
## Dynamic-range Histogram Self-Attention (DHSA)
class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)

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
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = x.clone()
        x[:, :c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, :c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, :c // 2] = out_replace
        return out


## Histogram Transformer Block (HTB)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.5, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x_out = x + self.ffn(self.norm_ff1(x))
        return x_out


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
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


# 使用HTB中的Dynamic-range Histogram Self-Attention替换 OSABlock中的attention
class PSABlock_DHSA(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention_histogram(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_DHSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock_DHSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


# 使用HTB替换 OSABlock
class C2PSA_HTB(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(TransformerBlock(self.c) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


##########################################################################
## 修正后的泰勒注意力机制 Taylor Attention
class TaylorAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(TaylorAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.scale = self.dim_head ** -0.5

        # QKV投影
        self.to_qkv = Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.to_qkv_dwconv = Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        # 输出投影
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 泰勒注意力的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 简化MSAR模块 - 避免复杂的维度操作
        self.correction_conv = nn.Conv2d(num_heads, num_heads, 3, padding=1, groups=num_heads)

    def forward(self, x):
        b, c, h, w = x.shape

        # 生成QKV
        qkv = self.to_qkv_dwconv(self.to_qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重塑为多头格式
        q = rearrange(q, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        k = rearrange(k, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
        v = rearrange(v, 'b (head d) h w -> b head (h w) d', head=self.num_heads)

        # 标准化
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 泰勒近似注意力计算
        # 预计算关键组件
        k_v = torch.einsum('b h n d, b h n c -> b h d c', k, v)  # K^T V
        k_sum = k.sum(dim=2)  # ∑K_j
        v_sum = v.sum(dim=2, keepdim=True)  # ∑V_j

        n = q.shape[2]  # 序列长度

        # 线性复杂度计算
        numerator = v_sum + torch.einsum('b h n d, b h d c -> b h n c', q, k_v) * self.scale
        denominator = n + torch.einsum('b h n d, b h d -> b h n', q, k_sum) * self.scale

        # 应用注意力
        out = numerator / (denominator.unsqueeze(-1) + 1e-6)

        # 简化的MSAR修正 - 直接应用卷积修正而不进行复杂的维度扩展
        # 重塑为空间格式进行卷积修正
        out_spatial = out.mean(dim=-1)  # [b, num_heads, h*w]
        out_spatial = out_spatial.reshape(b, self.num_heads, h, w)
        correction = self.correction_conv(out_spatial)

        # 将修正应用到原始输出
        correction = correction.reshape(b, self.num_heads, n, 1)
        out = out + correction * 0.1  # 使用较小的权重避免过度修正

        # 重塑回原始格式
        out = rearrange(out, 'b head (h w) d -> b (head d) h w', h=h, w=w)
        return self.project_out(out)


##########################################################################
## 改进的混合注意力: 泰勒注意力 + 直方图注意力
class HybridAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, fusion_ratio=0.5):
        super(HybridAttention, self).__init__()

        # 确保两个注意力模块的num_heads一致
        self.taylor_attn = TaylorAttention(dim, num_heads, bias)
        self.histogram_attn = Attention_histogram(dim, num_heads, bias, True)

        # 改进的融合权重 - 使用可学习的参数
        self.fusion_weights = nn.Parameter(torch.ones(2))

        # 使用简单加权融合，避免复杂门控
        self.use_adaptive_fusion = False

    def forward(self, x):
        taylor_out = self.taylor_attn(x)
        histogram_out = self.histogram_attn(x)

        # 简单加权融合
        weights = F.softmax(self.fusion_weights, dim=0)
        out = weights[0] * taylor_out + weights[1] * histogram_out

        return out


##########################################################################
## 泰勒Transformer块
class TaylorTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2.5, bias=False,
                 LayerNorm_type='WithBias', attention_type='taylor'):
        super(TaylorTransformerBlock, self).__init__()

        # 选择注意力类型
        if attention_type == 'taylor':
            self.attn = TaylorAttention(dim, num_heads, bias)
        elif attention_type == 'hybrid':
            self.attn = HybridAttention(dim, num_heads, bias)
        else:  # 原有直方图注意力
            self.attn = Attention_histogram(dim, num_heads, bias, True)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        # FeedForward网络 (保持原有DGFF结构)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # 注意力部分
        x = x + self.attn(self.norm1(x))
        # FFN部分
        x = x + self.ffn(self.norm2(x))
        return x


##########################################################################
## 多尺度泰勒注意力块
class MultiScaleTaylorBlock(nn.Module):
    def __init__(self, dim, num_heads=8, scales=[1, 2, 4], bias=False):
        super(MultiScaleTaylorBlock, self).__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            TaylorAttention(dim, num_heads, bias) for _ in scales
        ])
        self.fusion = Conv2d(dim * len(scales), dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        outputs = []

        for scale, attn in zip(self.scales, self.attentions):
            if scale > 1:
                # 下采样
                x_scaled = F.interpolate(x, scale_factor=1 / scale, mode='bilinear')
                out_scaled = attn(x_scaled)
                # 上采样回原尺寸
                out_scaled = F.interpolate(out_scaled, size=(h, w), mode='bilinear')
            else:
                out_scaled = attn(x)

            outputs.append(out_scaled)

        # 融合多尺度结果
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)


##########################################################################
## 使用泰勒注意力的PSA块
class PSABlock_Taylor(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=8, shortcut=True, attention_type='taylor'):
        super().__init__()
        self.add = shortcut

        # 使用泰勒注意力
        if attention_type == 'taylor':
            self.attn = TaylorAttention(c, num_heads)
        elif attention_type == 'multiscale_taylor':
            self.attn = MultiScaleTaylorBlock(c, num_heads)
        else:  # hybrid
            self.attn = HybridAttention(c, num_heads)

        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


##########################################################################
## 使用泰勒注意力的C2PSA模块
class C2PSA_Taylor(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, attention_type='taylor'):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 使用泰勒Transformer块
        self.m = nn.Sequential(*[
            TaylorTransformerBlock(self.c, attention_type=attention_type)
            for _ in range(n)
        ])

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


##########################################################################
## 改进的性能对比测试
if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 输入张量
    batch_size = 2
    input_tensor = torch.randn(batch_size, 256, 64, 64)

    print("=== 性能对比测试 ===")


    # 参数数量计算函数
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    try:
        # 1. 原有HTB
        htb = TransformerBlock(256)
        output_htb = htb(input_tensor)
        print(f"✓ HTB - Input: {input_tensor.shape}, Output: {output_htb.shape}")

        # 2. 泰勒注意力HTB
        taylor_htb = TaylorTransformerBlock(256, attention_type='taylor')
        output_taylor = taylor_htb(input_tensor)
        print(f"✓ Taylor HTB - Input: {input_tensor.shape}, Output: {output_taylor.shape}")

        # 3. 混合注意力HTB
        hybrid_htb = TaylorTransformerBlock(256, attention_type='hybrid')
        output_hybrid = hybrid_htb(input_tensor)
        print(f"✓ Hybrid HTB - Input: {input_tensor.shape}, Output: {output_hybrid.shape}")

        # 4. 多尺度泰勒注意力
        multiscale_taylor = MultiScaleTaylorBlock(256)
        output_multiscale = multiscale_taylor(input_tensor)
        print(f"✓ MultiScale Taylor - Input: {input_tensor.shape}, Output: {output_multiscale.shape}")

        # 5. C2PSA泰勒版本
        c2psa_taylor = C2PSA_Taylor(256, 256, n=2, attention_type='taylor')
        output_c2psa = c2psa_taylor(input_tensor)
        print(f"✓ C2PSA Taylor - Input: {input_tensor.shape}, Output: {output_c2psa.shape}")

        # 参数数量对比
        print(f"\n=== 参数数量对比 ===")
        print(f"HTB 参数数量: {count_parameters(htb):,}")
        print(f"Taylor HTB 参数数量: {count_parameters(taylor_htb):,}")
        print(f"Hybrid HTB 参数数量: {count_parameters(hybrid_htb):,}")
        print(f"MultiScale Taylor 参数数量: {count_parameters(multiscale_taylor):,}")
        print(f"C2PSA Taylor 参数数量: {count_parameters(c2psa_taylor):,}")

        # 计算复杂度分析
        print(f"\n=== 计算复杂度分析 ===")
        seq_len = 64 * 64
        print(f"序列长度: {seq_len}")
        print(f"标准Softmax注意力: O(n²) = O({seq_len}²) = {seq_len ** 2:,}")
        print(f"泰勒注意力: O(n) = O({seq_len}) = {seq_len:,}")
        print(f"加速比: {seq_len ** 2 / seq_len:.1f}x")

        # GPU内存使用测试
        if torch.cuda.is_available():
            device = torch.device('cuda')
            input_tensor = input_tensor.to(device)
            taylor_htb = taylor_htb.to(device)

            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
            output = taylor_htb(input_tensor)
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated()

            print(f"\n✓ GPU内存使用: {(end_memory - start_memory) / 1024 ** 2:.2f} MB")

        print("\n✓ 所有测试通过！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()