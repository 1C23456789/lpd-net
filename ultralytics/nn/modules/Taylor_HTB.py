import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
        sigma = torch.clamp(sigma, min=1e-8)
        return x / torch.sqrt(sigma)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        sigma = torch.clamp(sigma, min=1e-8)
        return (x - mu) / torch.sqrt(sigma)


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

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
## Dynamic-range Histogram Self-Attention (DHSA) with Taylor Expansion
class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 可学习的泰勒展开系数（加 clamp 防止极端值）
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(-0.05))

        self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def pad(self, x, factor):
        hw = x.shape[-1]
        pad_len = (factor - hw % factor) % factor
        if pad_len == 0:
            return x, [0, 0]
        x = F.pad(x, [0, pad_len], 'constant', 0)
        return x, [0, pad_len]

    def unpad(self, x, t_pad):
        if t_pad[1] == 0:
            return x
        return x[..., :-t_pad[1]]

    def taylor_attention_score(self, q, k):
        """
        Compute attention scores using Taylor-inspired expansion:
        score = dot + alpha * dot^2 + beta * ||q - k||^2
        """
        B, H, N, D = q.shape
        scale = D ** -0.5  # 标准缩放，防止点积过大

        dot = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]
        dot_sq = dot ** 2

        q_norm_sq = q.pow(2).sum(dim=-1, keepdim=True)      # [B, H, N, 1]
        k_norm_sq = k.pow(2).sum(dim=-1, keepdim=True)      # [B, H, N, 1]
        diff_norm_sq = q_norm_sq + k_norm_sq.transpose(-2, -1) - 2 * dot * (D ** 0.5)

        # 限制参数范围（防止训练发散）
        alpha = torch.clamp(self.alpha, -1.0, 1.0)
        beta = torch.clamp(self.beta, -1.0, 1.0)

        score = dot + alpha * dot_sq + beta * diff_norm_sq
        return score

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, _ = self.pad(k, self.factor)
        v, _ = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor

        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"

        q = rearrange(q, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)

        attn_scores = self.taylor_attention_score(q, k) * self.temperature
        attn = F.softmax(attn_scores, dim=-1)  # ✅ 数值稳定的 softmax

        out = torch.matmul(attn, v)
        out = rearrange(out, f'{shape_tar} -> {shape_ori}', factor=self.factor, hw=hw, b=b, head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape

        # === 创新：对前半通道做直方图排序 ===
        x_sort, idx_h = x[:, :c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = x.clone()
        x[:, :c // 2] = x_sort

        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        # === 对 v 排序，并同步 q/k ===
        v_flat = v.view(b, c, -1)
        v_sorted, idx = v_flat.sort(dim=-1)

        # 获取逆序索引（关键修复！）
        idx_inv = torch.argsort(idx, dim=-1)

        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v_sorted, True)
        out2 = self.reshape_attn(q2, k2, v_sorted, False)

        # === 用 idx_inv 恢复原始空间顺序 ===
        out1 = torch.gather(out1, dim=2, index=idx_inv).view(b, c, h, w)
        out2 = torch.gather(out2, dim=2, index=idx_inv).view(b, c, h, w)

        out = out1 * out2

        out = self.project_out(out)

        # === 恢复前半通道的原始排序（非 in-place 方式）===
        out_front = out[:, :c // 2]  # 前半通道
        out_back = out[:, c // 2:]  # 后半通道（未排序，保持不变）

        # 恢复 width 排序
        idx_w_inv = torch.argsort(idx_w, dim=-1)
        out_front = torch.gather(out_front, dim=-1, index=idx_w_inv)
        # 恢复 height 排序
        idx_h_inv = torch.argsort(idx_h, dim=-2)
        out_front = torch.gather(out_front, dim=-2, index=idx_h_inv)

        # 拼接回完整通道（关键：避免 in-place 赋值）
        out = torch.cat([out_front, out_back], dim=1)
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


# 使用 HTB 中的 DHSA 替换 PSA Block
class PSABlock_Taylor_DHSA(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention_histogram(c, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_Taylor_DHSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock_Taylor_DHSA(self.c, num_heads=max(1, self.c // 64)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA_Taylor_HTB(nn.Module):
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


if __name__ == '__main__':
    # 测试数值稳定性
    torch.manual_seed(42)
    TB = TransformerBlock(256)
    x = torch.randn(2, 256, 64, 64, requires_grad=True)
    out = TB(x)
    loss = out.mean()
    loss.backward()

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Has NaN in output:", torch.isnan(out).any().item())
    print("Has NaN in input grad:", torch.isnan(x.grad).any().item())