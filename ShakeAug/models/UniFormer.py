# All rights reserved.
from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import _cfg
from timm.models import register_model
from timm.layers import DropPath, trunc_normal_, to_2tuple

layer_scale = False
init_value = 1e-6


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 drop=0., drop_path=0.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = CMlp(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.transpose(1, 2).reshape(B, C, H, W)


class PatchEmbed(nn.Module):
    """Dynamic Patch Embedding (no fixed image size)"""
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x.transpose(1, 2).reshape(B, C, H, W)


class UniFormer(nn.Module):
    def __init__(self, depth, embed_dim, head_dim,
                 num_classes=1000, mlp_ratio=4.,
                 drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.patch_embed1 = PatchEmbed(4, 3, embed_dim[0])
        self.patch_embed2 = PatchEmbed(2, embed_dim[0], embed_dim[1])
        self.patch_embed3 = PatchEmbed(2, embed_dim[1], embed_dim[2])
        self.patch_embed4 = PatchEmbed(2, embed_dim[2], embed_dim[3])

        dpr = torch.linspace(0, drop_path_rate, sum(depth)).tolist()
        num_heads = [d // head_dim for d in embed_dim]

        self.blocks1 = nn.ModuleList([
            CBlock(embed_dim[0], num_heads[0], mlp_ratio, drop_rate, dpr[i])
            for i in range(depth[0])
        ])
        self.blocks2 = nn.ModuleList([
            CBlock(embed_dim[1], num_heads[1], mlp_ratio, drop_rate, dpr[i + depth[0]])
            for i in range(depth[1])
        ])
        self.blocks3 = nn.ModuleList([
            SABlock(embed_dim[2], num_heads[2], mlp_ratio, drop_rate,
                    dpr[i + depth[0] + depth[1]])
            for i in range(depth[2])
        ])
        self.blocks4 = nn.ModuleList([
            SABlock(embed_dim[3], num_heads[3], mlp_ratio, drop_rate,
                    dpr[i + depth[0] + depth[1] + depth[2]])
            for i in range(depth[3])
        ])

        self.norm = nn.BatchNorm2d(embed_dim[-1])
        self.head = nn.Linear(embed_dim[-1], num_classes)

    def forward(self, x):
        x = self.patch_embed1(x)
        for b in self.blocks1: x = b(x)
        x = self.patch_embed2(x)
        for b in self.blocks2: x = b(x)
        x = self.patch_embed3(x)
        for b in self.blocks3: x = b(x)
        x = self.patch_embed4(x)
        for b in self.blocks4: x = b(x)
        x = self.norm(x)
        return self.head(x.flatten(2).mean(-1))


# ====== Factory functions (RESTORED) ======

@register_model
def uniformer_small(num_classes=1000, **kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        num_classes=num_classes
    )
    model.default_cfg = _cfg()
    return model


@register_model
def uniformer_base(num_classes=1000, **kwargs):
    model = UniFormer(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        num_classes=num_classes
    )
    model.default_cfg = _cfg()
    return model


@register_model
def uniformer_base_ls(num_classes=1000, **kwargs):
    global layer_scale
    layer_scale = True
    return uniformer_base(num_classes)
