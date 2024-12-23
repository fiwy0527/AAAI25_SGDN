# @Author:Fangwenxuan
from typing import Union, Tuple
import kornia
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import numbers


# from basicsr.utils.registry import ARCH_REGISTRY


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


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

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='withBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


#  following https://github.com/zcablii/LSKNet
class LSKAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        hidden = self.to_hidden(x)
        cont_x = self.to_hidden_dw(hidden)
        x1, x2 = cont_x.chunk(2, dim=1)

        attn1 = self.conv0(x1)
        attn2 = self.conv_spatial(x2)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1, groups=in_features),
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, groups=out_features),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class ConvLayer(nn.Module):
    def __init__(self, net_depth, dim):
        super().__init__()
        self.dim = dim

        self.net_depth = net_depth
        self.attention = LSKAttention(self.dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.net_depth) ** (
                    -1 / 4)  # self.net_depth ** (-1/2), the deviation seems to be too small, a bigger one may be better
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.attention(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, net_depth, dim, conv_layer=ConvLayer,
                 mlp_ratio=4.0, drop_path=0., ):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.token_mixer = conv_layer(net_depth, dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gdfn = Mlp(network_depth=net_depth, in_features=dim,
                        hidden_features=mlp_hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.gdfn(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, net_depth, dim, index, depth, layers,
                 conv_layer=ConvLayer,
                 drop_path_rate=0., ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        block_dpr = 0
        for block_idx in range(depth):
            block_dpr = drop_path_rate * (
                    block_idx + sum(layers[:index])) / (sum(layers) - 1)
        # build blocks
        self.blocks = nn.ModuleList([
            BasicBlock(net_depth, dim, conv_layer,
                       drop_path=block_dpr)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class CAB(nn.Module):
    def __init__(self, dim, num_heads=8, bias=True):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention
class IAM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(IAM, self).__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim, LayerNorm_type='withBisa')
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


# Interaction Attention Module
class IAMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        self.proj = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        self.rgb_cab = IAM(self.dim)
        self.ycbcr_cab = IAM(self.dim)

    def forward(self, in_feats):
        out_rgb = self.rgb_cab(in_feats[0], in_feats[1])
        out_ycbcr = self.ycbcr_cab(in_feats[1], in_feats[0])

        x = self.proj(out_ycbcr + out_rgb)
        return x, out_rgb, out_ycbcr


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class MDHTAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, bias=True):
        super(MDHTAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=bias)
        self.act = nn.Tanh()

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, bias=False):
        super(SpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, out_dim, kernel_size=1, bias=bias)
        self.act = nn.Tanh()

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q, k = F.normalize(q, dim=-2), F.normalize(k, dim=-2)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = attn.softmax(dim=-1) @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Phase integration module
class PIM(nn.Module):
    def __init__(self, channel):
        super(PIM, self).__init__()

        self.processmag = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0))

        self.processpha = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 1, 1, 0))

    def forward(self, rgb_x, ycbcr_x):
        rgb_fft = torch.fft.rfft2(rgb_x, norm='backward')
        ycbcr_fft = torch.fft.rfft2(ycbcr_x, norm='backward')
        rgb_amp = torch.abs(rgb_fft)
        rgb_phase = torch.angle(rgb_fft)

        ycbcr_amp = torch.abs(ycbcr_fft)
        ycbcr_phase = torch.angle(ycbcr_fft)

        rgb_amp = self.processmag(rgb_amp)
        rgb_phase = self.processmag(rgb_phase)

        ycbcr_amp = self.processmag(ycbcr_amp)
        ycbcr_phase = self.processmag(ycbcr_phase)

        mix_phase = rgb_phase + ycbcr_phase

        out_rgb = torch.fft.irfft2(rgb_amp * torch.exp(1j * mix_phase), norm='backward')
        out_ycbcr = torch.fft.irfft2(ycbcr_amp * torch.exp(1j * mix_phase), norm='backward')
        return out_rgb, out_ycbcr


class BI_color_Guidance_Bridge(nn.Module):
    def __init__(self, dim):
        super(BI_color_Guidance_Bridge, self).__init__()

        self.detail_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.smooth_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.iam = IAMB(dim)
        self.detail_attention = SpatialAttention(dim, dim)

        self.color_attention = MDHTAttention(dim, dim)

        self.pim = PIM(dim)

        self.pixel_shuffle = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.rgb_feat_conv = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1, stride=2, groups=dim // 2),
            nn.Softmax(dim=1)
        )
        self.ycbcr_feat_conv = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=3, padding=1, stride=2, groups=dim // 2),
            nn.Softmax(dim=1)
        )

    def forward(self, in_feats, pre_down_list=None):
        down_feat0 = self.smooth_pool(in_feats[0])
        down_feat1 = self.detail_pool(in_feats[1])

        if pre_down_list is not None:
            down_feat0 = self.color_attention(down_feat0 * self.rgb_feat_conv(pre_down_list[0]))
            down_feat1 = self.detail_attention(down_feat1 * self.ycbcr_feat_conv(pre_down_list[1]))
        else:
            down_feat0, down_feat1 = self.pim(down_feat0, down_feat1)

        feat0 = self.pixel_shuffle(down_feat0)
        feat1 = self.pixel_shuffle(down_feat1)
        inp_fusion_out, rgb_feat, ycbcr_feat = self.iam([feat0, feat1])

        return inp_fusion_out, rgb_feat, ycbcr_feat, down_feat0, down_feat1


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(feats_sum)
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


# COLOR ENHANCEMENT MODULE
class CEM(nn.Module):
    def __init__(self, channel):
        super(CEM, self).__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x1_amp1 = torch.mean(x, dim=1, keepdim=True)
        x1_amp1 = x - x1_amp1
        att = self.adaptive_avg_pool(x1_amp1)
        att = F.softmax(att, 1)
        return att


# @ARCH_REGISTRY.register()
class SGDN(nn.Module):
    def __init__(self, base_dim=32, encoder_depths=[4, 4, 6, 8], conv_layer=ConvLayer,
                 decoder_depths=[8, 6, 4, 4], fusion_layer=SKFusion):
        super(SGDN, self).__init__()
        # setting
        self.encoder_num = len(encoder_depths)
        net_depth = sum(encoder_depths)
        encoder_dim = [2 ** i * base_dim for i in range(len(encoder_depths))]

        # input convolution
        self.inconv = PatchEmbed(patch_size=1, in_chans=3, embed_dim=encoder_dim[0], kernel_size=3)

        # backbone
        self.layers = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.sDiffPropagator = nn.ModuleList()

        self.decoder4 = BasicLayer(dim=encoder_dim[-1], depth=2, index=0, layers=decoder_depths,
                                   net_depth=8,
                                   conv_layer=conv_layer)

        self.up4 = PatchUnEmbed(patch_size=2, out_chans=encoder_dim[2], embed_dim=encoder_dim[-1])

        self.decoder3 = BasicLayer(dim=encoder_dim[2], depth=2, index=1, layers=decoder_depths,
                                   net_depth=8,
                                   conv_layer=conv_layer)

        self.up3 = PatchUnEmbed(patch_size=2, out_chans=encoder_dim[1], embed_dim=encoder_dim[2])

        self.decoder2 = BasicLayer(dim=encoder_dim[1], depth=2, index=2, layers=decoder_depths,
                                   net_depth=8,
                                   conv_layer=conv_layer)

        self.up2 = PatchUnEmbed(patch_size=2, out_chans=base_dim, embed_dim=encoder_dim[1])

        self.decoder1 = BasicLayer(dim=base_dim, depth=2, index=3, layers=decoder_depths,
                                   net_depth=8,
                                   conv_layer=conv_layer)

        for i in range(self.encoder_num):
            self.layers.append(
                BasicLayer(dim=encoder_dim[i], depth=encoder_depths[i], index=i, layers=encoder_depths,
                           net_depth=net_depth,
                           conv_layer=conv_layer))
            self.sDiffPropagator.append(BI_color_Guidance_Bridge(encoder_dim[i]))
        # 删除最后一个模块
        del self.sDiffPropagator[-1]

        for i in range(self.encoder_num - 1):
            self.downs.append(PatchEmbed(patch_size=2, in_chans=encoder_dim[i], embed_dim=encoder_dim[i + 1]))
            self.skips.append(nn.Conv2d(encoder_dim[i], encoder_dim[i], 1))
            self.fusions.append(fusion_layer(encoder_dim[i]))

        # output convolution
        self.outconv = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=encoder_dim[0], kernel_size=3)
        self.outconv128 = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=64, kernel_size=3)
        self.outconv64 = PatchUnEmbed(patch_size=1, out_chans=3, embed_dim=128, kernel_size=3)
        self.cem = CEM(256)

    def forward(self, x):
        ycbcr_img = kornia.color.rgb_to_ycbcr(x)

        feat = self.inconv(x)
        ycbcr_feat = self.inconv(ycbcr_img)

        ycbcr_feat_list = []
        feat_list = []
        sdiff_feat_list = []

        feat = self.layers[0](feat)
        ycbcr_feat = self.layers[0](ycbcr_feat)

        feat_list.append(feat)
        ycbcr_feat_list.append(ycbcr_feat)
        sdiff_feat, rgb_mid_feat, ycbcr_mid_feat, down_rgb_feat0, down_ycbcr_feat0 = self.sDiffPropagator[0](
            [feat, ycbcr_feat])
        sdiff_feat_list.append(sdiff_feat)
        feat = self.downs[0](feat + rgb_mid_feat)
        ycbcr_feat = self.downs[0](ycbcr_feat + ycbcr_mid_feat)

        feat = self.layers[1](feat)
        ycbcr_feat = self.layers[1](ycbcr_feat)

        feat_list.append(feat)
        ycbcr_feat_list.append(ycbcr_feat)
        sdiff_feat, rgb_mid_feat, ycbcr_mid_feat, down_rgb_feat1, down_ycbcr_feat1 = self.sDiffPropagator[1](
            [feat, ycbcr_feat], [down_rgb_feat0, down_ycbcr_feat0])
        sdiff_feat_list.append(sdiff_feat)
        feat = self.downs[1](feat + rgb_mid_feat)
        ycbcr_feat = self.downs[1](ycbcr_feat + ycbcr_mid_feat)

        feat = self.layers[2](feat)
        ycbcr_feat = self.layers[2](ycbcr_feat)
        feat_list.append(feat)
        ycbcr_feat_list.append(ycbcr_feat)

        sdiff_feat, rgb_mid_feat, ycbcr_mid_feat, down_rgb_feat2, down_ycbcr_feat2 = self.sDiffPropagator[2](
            [feat, ycbcr_feat], [down_rgb_feat1, down_ycbcr_feat1])

        sdiff_feat_list.append(sdiff_feat)
        feat = self.downs[2](feat + rgb_mid_feat)
        ycbcr_feat = self.downs[2](ycbcr_feat + ycbcr_mid_feat)

        feat = self.layers[-1](feat)
        ycbcr_feat = self.layers[-1](ycbcr_feat)
        feat_list.append(feat)
        ycbcr_feat_list.append(ycbcr_feat)
        cem_f = self.cem(ycbcr_feat)

        feat_decoder_input = feat_list[-1] * cem_f
        feat_decoder_input = feat_decoder_input + ycbcr_feat

        out = self.decoder4(feat_decoder_input)
        out = self.up4(out)
        out = self.fusions[2]([self.skips[2](sdiff_feat_list[2]), out])
        out = self.decoder3(out)
        out_64 = self.outconv64(out)
        out = self.up3(out)
        out = self.fusions[1]([self.skips[1](sdiff_feat_list[1]), out])
        out = self.decoder2(out)
        out_128 = self.outconv128(out)
        out = self.up2(out)
        out = self.fusions[0]([self.skips[0](sdiff_feat_list[0]), out])
        out = self.decoder1(out)
        degraded_img = self.outconv(out) + x

        return degraded_img, out_128, out_64


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    new_h = h + (patch_size - h % patch_size) % patch_size
    new_w = w + (patch_size - w % patch_size) % patch_size
    pad_h = new_h - h
    pad_w = new_w - w
    x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
    return x


if __name__ == '__main__':
    # from torchstat import stat
    import os
    from thop import profile
    from ptflops import get_model_complexity_info

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    net = SGDN().cuda().eval()
    # print(net)
    input = torch.rand([1, 3, 128, 128]).cuda()
    # source_img = pad_img(input, 16)
    print(net(input))
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(net, (3, 256, 256), as_strings=True,
                                                  print_per_layer_stat=True,
                                                  verbose=True)
        print('Flops:  ', flops)
        print('Params: ', params)
