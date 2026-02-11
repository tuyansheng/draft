import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models import register_model
from functools import partial
from timm.models.vision_transformer import _cfg


class OffsetGenerator:
    @classmethod
    def build_qk_vec(cls, n_patch_side: int, pad_size: int, device, dtype):
        grid_1d = torch.linspace(-1, 1, n_patch_side, device=device, dtype=dtype)

        if pad_size > 0:
            step = grid_1d[-1] - grid_1d[-2]
            pad_dist = torch.cumsum(step.repeat(pad_size), dim=0)
            grid_1d = torch.cat([(-1 - pad_dist).flip(dims=[0]), grid_1d, 1 + pad_dist])
            n_patch_side += (pad_size * 2)

        n_tokens = n_patch_side ** 2

        grid_y = grid_1d.view(-1, 1).repeat(1, n_patch_side)
        grid_x = grid_1d.view(1, -1).repeat(n_patch_side, 1)
        grid = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2)  # (n_tokens, 2)

        grid_q = grid.view(-1, 1, 2).repeat(1, n_tokens, 1)       # (n_tokens, n_tokens, 2)
        grid_k = grid.view(1, -1, 2).repeat(n_tokens, 1, 1)       # (n_tokens, n_tokens, 2)

        qk_vec = grid_k - grid_q                                  # (n_tokens, n_tokens, 2)
        return qk_vec


class PeripheralPositionEncoding(nn.Module):
    def __init__(self, num_heads, norm_init, kernel_size=3):
        super().__init__()
        in_channel = hid_channel = num_heads * 4
        out_channel = num_heads
        self.remove_pad = (kernel_size // 2) * 2
        self.norm_init = norm_init

        self.pad_size = 0
        self.conv1 = nn.Conv2d(in_channel, hid_channel, kernel_size=kernel_size, stride=1, padding=self.pad_size, bias=True)
        self.conv2 = nn.Conv2d(hid_channel, out_channel, kernel_size=kernel_size, stride=1, padding=self.pad_size, bias=True)
        self.gn1 = nn.GroupNorm(hid_channel, hid_channel)
        self.gn2 = nn.GroupNorm(out_channel, out_channel)

        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Sigmoid()

        self.apply(self._init_weights)
        self.gn2.apply(self._peripheral_init)

    def _peripheral_init(self, m):
        if isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, float(self.norm_init[0]))
            nn.init.constant_(m.weight, float(self.norm_init[1]))

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.constant_(m.weight, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (N, N, D_r) where N = side^2, D_r = num_heads*4
        n_token = x.size(1)
        side = int(math.sqrt(n_token))
        if side * side != n_token:
            raise RuntimeError(f"PPE: token count must be square, got N={n_token}")

        # (i j) (k l) d -> d i j k l
        x = rearrange(x, '(i j) (k l) d -> d i j k l', i=side, j=side, k=side, l=side)
        if self.remove_pad > 0:
            x = x[:, self.remove_pad:-self.remove_pad, self.remove_pad:-self.remove_pad, ...]
        # d i j k l -> (i j) d k l
        x = rearrange(x, 'd i j k l -> (i j) d k l')

        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.gn2(x)

        side2 = side - (self.remove_pad * 2)
        y = rearrange(x, '(i j) h k l -> () h (i j) (k l)', i=side2, j=side2)
        y = self.act2(y)  # (1, H, N, N)
        return y


class MPA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_init=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.activation = PeripheralPositionEncoding(num_heads, norm_init)
        self.exp = lambda x: torch.exp(x - torch.max(x, -1, keepdim=True)[0])

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # cache: key=(device, dtype, N) -> weight(1,H,N,N)
        self._weight_cache = {}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def build_weight(self, rpe_embed, N, device, dtype):
        # rpe_embed: (N, N, D_r)
        w = self.activation(rpe_embed)  # expected (1,H,N,N)
        if w.dim() != 4:
            raise RuntimeError(f"MPA: weight must be 4D, got {tuple(w.shape)}")
        if w.shape[1] != self.num_heads or w.shape[2] != N or w.shape[3] != N:
            raise RuntimeError(f"MPA: bad weight shape {tuple(w.shape)}, expected (1,{self.num_heads},{N},{N})")
        return w.to(device=device, dtype=dtype).contiguous()

    def forward(self, x, rpe_embed):
        # x: (B, N, C)
        B, N, _ = x.shape

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,N,N)
        dots = self.exp(dots)

        # weight: (1,H,N,N) from cache; if not provided (rpe_embed None) -> must exist
        if rpe_embed is not None:
            key = (x.device, x.dtype, N)
            w = self._weight_cache.get(key, None)
            if w is None:
                w = self.build_weight(rpe_embed, N=N, device=x.device, dtype=x.dtype)
                self._weight_cache[key] = w
        else:
            key = (x.device, x.dtype, N)
            w = self._weight_cache.get(key, None)
            if w is None:
                raise RuntimeError("MPA: rpe_embed is None but no cached weight found. Call forward once with rpe_embed.")

        attn = w * dots
        attn = F.normalize(attn, p=1, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features if out_features is not None else in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CPE(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        side = int(math.sqrt(N))
        if side * side != N:
            raise RuntimeError(f"CPE: token count must be square, got N={N}")

        x = rearrange(x, 'b (i j) d -> b d i j', i=side, j=side)
        x = self.proj(x)
        x = rearrange(x, 'b d i j -> b (i j) d', i=side, j=side)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_init=None, next_dim=None, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.stage_change = dim != next_dim
        self.res_path = nn.Linear(dim, next_dim) if self.stage_change else nn.Identity()

        self.attn = MPA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, norm_init=norm_init, **kwargs
        )
        self.conv = CPE(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=next_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rel_pos_embed):
        x = x + self.drop_path(self.attn(self.norm1(self.conv(x)), rel_pos_embed))
        x = self.res_path(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, use_pos_embed=False, emb_dims=None, convstem_dims=None,
                 rpe_kernel_size=3):
        super().__init__()

        self.num_heads = num_heads
        self.rpe_kernel_size = rpe_kernel_size

        self.conv_layers = ConvolutionalStem(convstem_dims)

        stages = [2, 2, 6, 2]
        ch_dims = []
        for dim, stage in zip(emb_dims, stages):
            ch_dims += [dim] * stage
        ch_dims += [ch_dims[-1]]

        self.num_classes = num_classes
        depth = len(ch_dims) - 1

        # rpe projection (shared, but rpe base depends on N)
        D_r = num_heads * 4
        self.rpe_proj = nn.Linear(1, D_r)
        self.rpe_proj.apply(self._peripheral_init)

        # rpe cache: key=(device, dtype, N) -> rpe_embed (N,N,D_r)
        self._rpe_cache = {}

        # Peripheral init schedule (keep same spirit, but device-aware at runtime)
        # store on CPU as float tensors; move per forward
        norm_bias = torch.linspace(-5.0, 4.0, steps=depth)
        norm_weight = torch.linspace(3.0, 0.01, steps=depth)
        self._norm_init_table = torch.stack([norm_bias, norm_weight]).t().contiguous()  # (depth, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=ch_dims[i], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                norm_init=self._norm_init_table[i], next_dim=ch_dims[i + 1]
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(ch_dims[-1])
        self.head = nn.Linear(ch_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.head.apply(self._init_weights)

    def _peripheral_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, -0.02)
            nn.init.constant_(m.bias, 0.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_rpe_embed(self, N: int, device, dtype):
        side = int(math.sqrt(N))
        if side * side != N:
            raise RuntimeError(f"RPE: token count must be square, got N={N}")

        pad_size = (self.rpe_kernel_size // 2) * 2

        qk_vec = OffsetGenerator.build_qk_vec(side, pad_size=pad_size, device=device, dtype=dtype)
        rpe = qk_vec.norm(p=2, dim=-1, keepdim=True)  # (Npad, Npad, 1) where Npad=(side+2*pad)^2
        # But PPE later crops internally, and expects input tokens to match original N in rearrange.
        # In original code, they used padded grid but still passed (Npad, Npad, Dr) into PPE which crops to N.
        # Here we must mimic: create padded Npad, and let PPE crop to N.
        rpe_embed = self.rpe_proj(rpe)  # (Npad, Npad, Dr)
        return rpe_embed

    def get_rpe_embed(self, N: int, device, dtype):
        key = (device, dtype, N)
        out = self._rpe_cache.get(key, None)
        if out is None:
            out = self._build_rpe_embed(N=N, device=device, dtype=dtype)
            self._rpe_cache[key] = out
        return out

    def forward_features(self, x):
        x = self.conv_layers(x)  # (B, N, C)
        B, N, _ = x.shape

        # build rpe for this N dynamically (CIFAR10 -> N=4)
        rpe_embed = self.get_rpe_embed(N=N, device=x.device, dtype=x.dtype)

        # ensure norm_init moved correctly: blocks stored CPU tensors, PPE init reads floats anyway
        for blk in self.blocks:
            x = blk(x, rpe_embed)

        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class ConvolutionalStem(nn.Module):
    def __init__(self, n_filter_list, kernel_sizes=[3, 3, 3, 3], strides=[2, 2, 2, 2]):
        super().__init__()
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(
                    in_channels=n_filter_list[i],
                    out_channels=n_filter_list[i + 1],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=kernel_sizes[i] // 2
                ),
                nn.BatchNorm2d(n_filter_list[i + 1]),
                nn.ReLU(inplace=True),
            ) for i in range(len(n_filter_list) - 1)]
        )

        self.conv1x1 = nn.Conv2d(
            in_channels=n_filter_list[-1],
            out_channels=n_filter_list[-1],
            stride=1,
            kernel_size=1,
            padding=0
        )
        self.flatten = Rearrange('b c h w -> b (h w) c')

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.conv1x1(x)
        x = self.flatten(x)
        return x


@register_model
def pervit_tiny(num_classes: int, pretrained=False, **kwargs):
    num_heads = 4
    kwargs['emb_dims'] = [128, 192, 224, 280]
    kwargs['convstem_dims'] = [3, 48, 64, 96, 128]

    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def pervit_small(num_classes: int, pretrained=False, **kwargs):
    num_heads = 8
    kwargs['emb_dims'] = [272, 320, 368, 464]
    kwargs['convstem_dims'] = [3, 64, 128, 192, 272]

    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def pervit_medium(num_classes: int, pretrained=False, **kwargs):
    num_heads = 12
    kwargs['emb_dims'] = [312, 468, 540, 684]
    kwargs['convstem_dims'] = [3, 64, 192, 256, 312]

    model = VisionTransformer(
        num_heads=num_heads,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
