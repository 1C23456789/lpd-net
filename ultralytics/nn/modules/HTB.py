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
#    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
#    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


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
        return x / torch.sqrt(sigma+1e-5) #* self.weight

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
        return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
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
##Dynamic-range Histogram Self-Attention (DHSA)
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
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias

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
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x

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


##Histogram Transformer Block (HTB)
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.5, bias=False, LayerNorm_type='WithBias'):## Other option 'BiasFree'
        super(TransformerBlock, self).__init__()
        self.attn_g = Attention_histogram(dim, num_heads, bias, True)
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x_out = x + self.ffn(self.norm_ff1(x))

        return x_out


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))



#使用HTB中的Dynamic-range HistogramSelf-Attentiion替换 OSABlock中的attention

class PSABlock_DHSA(nn.Module):

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention_histogram(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_DHSA(nn.Module):

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock_DHSA(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))



#使用HTB替换 OSABlock

class C2PSA_HTB(nn.Module):

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(TransformerBlock(self.c) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


if __name__ == '__main__':
    TB = TransformerBlock(256)
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =TB(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)




# import numbers
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
#
# from .block import  PSABlock, C2PSA
#
# Conv2d = nn.Conv2d
#
# ## Layer Norm
# def to_2d(x):
#     return rearrange(x, 'b c h w -> b (h w c)')
#
# def to_3d(x):
# #    return rearrange(x, 'b c h w -> b c (h w)')
#     return rearrange(x, 'b c h w -> b (h w) c')
#
# def to_4d(x,h,w):
# #    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
#     return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
#
#
# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) #* self.weight
#
# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)
#
#         assert len(normalized_shape) == 1
#
#         self.normalized_shape = normalized_shape
#
#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) #* self.weight + self.bias
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type="WithBias"):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)
#
#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)
# ##########################################################################
# ## Dual-scale Gated Feed-Forward Network (DGFF)
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()
#
#         hidden_features = int(dim * ffn_expansion_factor)
#
#         self.project_in = Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#
#         self.dwconv_5 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2,
#                                groups=hidden_features // 4, bias=bias)
#         self.dwconv_dilated2_1 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2,
#                                         groups=hidden_features // 4, bias=bias, dilation=2)
#         self.p_unshuffle = nn.PixelUnshuffle(2)
#         self.p_shuffle = nn.PixelShuffle(2)
#
#         self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         x = self.project_in(x)
#         x = self.p_shuffle(x)
#         x1, x2 = x.chunk(2, dim=1)
#         x1 = self.dwconv_5(x1)
#         x2 = self.dwconv_dilated2_1(x2)
#         x = F.mish(x2) * x1
#         x = self.p_unshuffle(x)
#         x = self.project_out(x)
#         return x
#
# ##########################################################################
# ##Dynamic-range Histogram Self-Attention (DHSA)
# class Attention_histogram(nn.Module):
#     def __init__(self, dim, num_heads=4, bias=False, ifBox=True):
#         super(Attention_histogram, self).__init__()
#         self.factor = num_heads
#         self.ifBox = ifBox
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
#
#         self.qkv = Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
#         self.qkv_dwconv = Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
#         self.project_out = Conv2d(dim, dim, kernel_size=1, bias=bias)
#
#     def pad(self, x, factor):
#         hw = x.shape[-1]
#         t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
#         x = F.pad(x, t_pad, 'constant', 0)
#         return x, t_pad
#
#     def unpad(self, x, t_pad):
#         _, _, hw = x.shape
#         return x[:, :, t_pad[0]:hw - t_pad[1]]
#
#     def softmax_1(self, x, dim=-1):
#         logit = x.exp()
#         logit = logit / (logit.sum(dim, keepdim=True) + 1)
#         return logit
#
#     def normalize(self, x):
#         mu = x.mean(-2, keepdim=True)
#         sigma = x.var(-2, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias
#
#     def reshape_attn(self, q, k, v, ifBox):
#         b, c = q.shape[:2]
#         q, t_pad = self.pad(q, self.factor)
#         k, t_pad = self.pad(k, self.factor)
#         v, t_pad = self.pad(v, self.factor)
#         hw = q.shape[-1] // self.factor
#         shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
#         shape_tar = "b head (c factor) hw"
#         q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
#         k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
#         v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = self.softmax_1(attn, dim=-1)
#         out = (attn @ v)
#         out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
#                         head=self.num_heads)
#         out = self.unpad(out, t_pad)
#         return out
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         x_sort, idx_h = x[:, :c // 2].sort(-2)
#         x_sort, idx_w = x_sort.sort(-1)
#         x = x.clone()
#         x[:, :c // 2] = x_sort
#         qkv = self.qkv_dwconv(self.qkv(x))
#         q1, k1, q2, k2, v = qkv.chunk(5, dim=1)  # b,c,x,x
#
#         v, idx = v.view(b, c, -1).sort(dim=-1)
#         q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
#         k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
#         q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
#         k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)
#
#         out1 = self.reshape_attn(q1, k1, v, True)
#         out2 = self.reshape_attn(q2, k2, v, False)
#
#         out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
#         out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
#         out = out1 * out2
#         out = self.project_out(out)
#         out_replace = out[:, :c // 2]
#         out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
#         out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
#         out[:, :c // 2] = out_replace
#         return out
#
#
# ##Histogram Transformer Block (HTB)
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads=4, ffn_expansion_factor=2.5, bias=False, LayerNorm_type='WithBias'):## Other option 'BiasFree'
#         super(TransformerBlock, self).__init__()
#         self.attn_g = Attention_histogram(dim, num_heads, bias, True)
#         self.norm_g = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#
#         self.norm_ff1 = LayerNorm(dim, LayerNorm_type)
#
#     def forward(self, x):
#         x = x + self.attn_g(self.norm_g(x))
#         x_out = x + self.ffn(self.norm_ff1(x))
#
#         return x_out
#
#
# #使用THB中的Dynamic-range HistogramSelf-Attentiion替换 OSABlock中的attention
# class PSABlock_DHSA(PSABlock):
#
#
#     def __init__(self, c, qk_dim =16 , pdim=32, shortcut=True) -> None:
#         """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
#         super().__init__( c)
#
#         self.attn = Attention_histogram(c)
#
#
#
# class C2PSA_DHSA(C2PSA):
#
#     def __init__(self, c1, c2, n=1, e=0.5):
#         """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
#         super().__init__(c1, c2)
#         assert c1 == c2
#         self.c = int(c1 * e)
#
#         self.m = nn.Sequential(*(PSABlock_DHSA(self.c, qk_dim =16 , pdim=32) for _ in range(n)))
#
#
#
# #使用THB替换 OSABlock
# class C2PSA_HTB(C2PSA):
#
#     def __init__(self, c1, c2, n=1, e=0.5):
#         """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
#         super().__init__(c1, c2)
#         assert c1 == c2
#         self.c = int(c1 * e)
#
#
#         self.m = nn.Sequential(*(TransformerBlock(self.c) for _ in range(n)))
#
# if __name__ == '__main__':
#     TB = TransformerBlock(256)
#     #创建一个输入张量
#     batch_size = 8
#     input_tensor=torch.randn(batch_size, 256, 64, 64 )
#     #运行模型并打印输入和输出的形状
#     output_tensor =TB(input_tensor)
#     print("Input shape:",input_tensor.shape)
#     print("0utput shape:",output_tensor.shape)