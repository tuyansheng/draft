# shake_aug.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch.utils.data import Dataset


OffsetSpec = Union[Tuple[int, int], str, None]  # (oy,ox) or "random" or "random_each_iter" or None


@dataclass(frozen=True)
class ShakeRecipe:
    """
    One augmentation recipe.

    N: number of "shake groups" (each group = 4 passes: R, L, D, U)
    beta: transport probability strength in p = 1 - exp(-beta * diff)
    alpha: boundary permeability (B-mode): for each cross-cell adjacent pixel pair,
           allow crossing with prob alpha (independent per pair)
    S: grid cell size (h, w)
    offset: None/"random"/"random_each_iter"/(oy,ox)
    p_apply: apply this recipe with probability p_apply, else return original
    """
    N: int
    beta: float
    alpha: float
    S: Tuple[int, int]
    offset: OffsetSpec = "random"
    p_apply: float = 1.0


# -----------------------------
# Helpers: dtype conversions
# -----------------------------
def _to_int16_255(x: torch.Tensor) -> torch.Tensor:
    """
    Convert input image to int16 in [0,255] if possible.
    Expect x in shape (C,H,W) or (B,C,H,W). If float, assume [0,1] and scale.
    """
    if x.dtype.is_floating_point:
        return (x * 255.0).round().clamp(0, 255).to(torch.int16)
    if x.dtype == torch.uint8:
        return x.to(torch.int16)
    return x.to(torch.int16).clamp(0, 255)


def _to_float01(xi: torch.Tensor) -> torch.Tensor:
    return (xi.to(torch.float32) / 255.0).clamp(0.0, 1.0)


# -----------------------------
# Stateless RNG (deterministic per-sample, batch-order independent)
# Portable version: xorshift32 implemented with int64 + 32-bit masking.
# -----------------------------
_INV_2P32 = 1.0 / float(1 << 32)
_MASK32 = (1 << 32) - 1  # 0xFFFFFFFF
_idx_cache: dict[tuple[str, int], torch.Tensor] = {}


def _get_idx_i64(device: torch.device, L: int) -> torch.Tensor:
    key = (str(device), int(L))
    t = _idx_cache.get(key, None)
    if t is None or t.device != device or t.numel() != L or t.dtype != torch.int64:
        t = torch.arange(L, device=device, dtype=torch.int64)
        _idx_cache[key] = t
    return t


def _xorshift32(x: torch.Tensor) -> torch.Tensor:
    # x is int64 but assumed already masked to 32-bit non-negative
    x = x & _MASK32
    x = x ^ ((x << 13) & _MASK32)
    x = x ^ (x >> 17)
    x = x ^ ((x << 5) & _MASK32)
    return x & _MASK32


def _stateless_uniform(
    seeds: torch.Tensor,            # (B,) int64
    tail_shape: Tuple[int, ...],
    *,
    key: int,
    device: torch.device,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Return U[0,1) of shape (B, *tail_shape), deterministic w.r.t seeds+key.
    Implemented using xorshift32, safe on Windows CPU builds.
    """
    if seeds.dtype != torch.int64:
        seeds = seeds.to(torch.int64)

    B = seeds.shape[0]
    L = 1
    for d in tail_shape:
        L *= int(d)

    idx = _get_idx_i64(device, L)  # (L,) int64

    # Mix seed, idx, key in 32-bit ring (values remain >=0 so >> is logical enough)
    # Keep multipliers within int64 safe range
    key_mix = (int(key) * 0x9E3779B9) & _MASK32  # 32-bit
    state = (seeds.to(device).view(B, 1) + idx.view(1, L) + key_mix) & _MASK32

    z = _xorshift32(state)  # (B,L) in [0,2^32)
    u = (z.to(torch.float64) * _INV_2P32).to(dtype)
    return u.view(B, *tail_shape)


# -----------------------------
# Grid boundary masks (B-mode alpha: per-edge independent)
# -----------------------------
def _boundary_mask_1d(length_edges: int, cell: int, offset: torch.Tensor, device) -> torch.Tensor:
    """
    For edges between j and j+1 (j=0..W-2), boundary is True if crossing between cells.
    boundary at (j+1) where (j+1 - offset) % cell == 0

    offset: (B,) int64
    return: (B, length_edges) bool
    """
    if cell <= 0:
        raise ValueError("cell size must be positive")
    j1 = torch.arange(1, length_edges + 1, device=device, dtype=torch.int64)  # 1..W-1
    b = ((j1.view(1, -1) - offset.view(-1, 1)) % int(cell)) == 0
    return b


def _allowed_mask_horiz(
    B: int, H: int, W: int,
    wcell: int, ox: torch.Tensor, alpha: float,
    seeds: torch.Tensor, key: int, device: torch.device
) -> torch.Tensor:
    """
    Allowed mask for horizontal edges between (y,x) and (y,x+1): (B,1,H,W-1)
    Within-cell edges always allowed.
    Boundary edges allowed with prob alpha (B-mode: per-edge independent).
    """
    boundary_1d = _boundary_mask_1d(W - 1, wcell, ox, device=device)          # (B, W-1)
    boundary = boundary_1d.view(B, 1, 1, W - 1).expand(B, 1, H, W - 1)        # (B,1,H,W-1)
    if alpha <= 0.0:
        return (~boundary)
    if alpha >= 1.0:
        return torch.ones((B, 1, H, W - 1), device=device, dtype=torch.bool)
    u = _stateless_uniform(seeds, (1, H, W - 1), key=key, device=device)      # (B,1,H,W-1)
    open_on_boundary = u < float(alpha)
    allowed = (~boundary) | (boundary & open_on_boundary)
    return allowed


def _allowed_mask_vert(
    B: int, H: int, W: int,
    hcell: int, oy: torch.Tensor, alpha: float,
    seeds: torch.Tensor, key: int, device: torch.device
) -> torch.Tensor:
    """
    Allowed mask for vertical edges between (y,x) and (y+1,x): (B,1,H-1,W)
    """
    boundary_1d = _boundary_mask_1d(H - 1, hcell, oy, device=device)          # (B, H-1)
    boundary = boundary_1d.view(B, 1, H - 1, 1).expand(B, 1, H - 1, W)        # (B,1,H-1,W)
    if alpha <= 0.0:
        return (~boundary)
    if alpha >= 1.0:
        return torch.ones((B, 1, H - 1, W), device=device, dtype=torch.bool)
    u = _stateless_uniform(seeds, (1, H - 1, W), key=key, device=device)      # (B,1,H-1,W)
    open_on_boundary = u < float(alpha)
    allowed = (~boundary) | (boundary & open_on_boundary)
    return allowed


# -----------------------------
# Shake passes (batched)
# -----------------------------
def _prob_from_diff(diff: torch.Tensor, beta: float) -> torch.Tensor:
    return 1.0 - torch.exp(-float(beta) * diff.to(torch.float32))


def _pass_right_batched(x, beta, allowed, seeds, key):
    left = x[:, :, :, :-1]
    right = x[:, :, :, 1:]
    cond = (left > right) & allowed
    diff = (left - right).clamp_min(0)
    p = _prob_from_diff(diff, beta)
    u = _stateless_uniform(seeds, (x.shape[1], x.shape[2], x.shape[3]-1), key=key, device=x.device)
    m = (u < p) & cond

    x_new = x.clone()
    mi = m.to(x.dtype)
    x_new[:, :, :, :-1].sub_(mi)   # ✅ 原地减，不会被覆盖
    x_new[:, :, :, 1: ].add_(mi)   # ✅ 原地加，会在重叠区正确累积
    return x_new


def _pass_left_batched(x, beta, allowed, seeds, key):
    a = x[:, :, :, 1:]
    b = x[:, :, :, :-1]
    cond = (a > b) & allowed
    diff = (a - b).clamp_min(0)
    p = _prob_from_diff(diff, beta)
    u = _stateless_uniform(seeds, (x.shape[1], x.shape[2], x.shape[3]-1), key=key, device=x.device)
    m = (u < p) & cond

    x_new = x.clone()
    mi = m.to(x.dtype)
    x_new[:, :, :, 1: ].sub_(mi)
    x_new[:, :, :, :-1].add_(mi)
    return x_new


def _pass_down_batched(x, beta, allowed, seeds, key):
    top = x[:, :, :-1, :]
    bot = x[:, :, 1:, :]
    cond = (top > bot) & allowed
    diff = (top - bot).clamp_min(0)
    p = _prob_from_diff(diff, beta)
    u = _stateless_uniform(seeds, (x.shape[1], x.shape[2]-1, x.shape[3]), key=key, device=x.device)
    m = (u < p) & cond

    x_new = x.clone()
    mi = m.to(x.dtype)
    x_new[:, :, :-1, :].sub_(mi)
    x_new[:, :, 1:,  :].add_(mi)
    return x_new


def _pass_up_batched(x, beta, allowed, seeds, key):
    a = x[:, :, 1:, :]
    b = x[:, :, :-1, :]
    cond = (a > b) & allowed
    diff = (a - b).clamp_min(0)
    p = _prob_from_diff(diff, beta)
    u = _stateless_uniform(seeds, (x.shape[1], x.shape[2]-1, x.shape[3]), key=key, device=x.device)
    m = (u < p) & cond

    x_new = x.clone()
    mi = m.to(x.dtype)
    x_new[:, :, 1:,  :].sub_(mi)
    x_new[:, :, :-1, :].add_(mi)
    return x_new


def apply_shake_recipe_batch(
    x: torch.Tensor,                 # (B,C,H,W)
    recipe: ShakeRecipe,
    *,
    seeds: torch.Tensor,             # (B,) int64
) -> torch.Tensor:
    """
    Batched shake application (vectorized).
    Deterministic per-sample w.r.t seeds. Batch order doesn't matter.
    Returns float tensor (B,C,H,W) in [0,1].
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x as (B,C,H,W), got {tuple(x.shape)}")
    device = x.device
    seeds = seeds.to(device=device, dtype=torch.int64)

    B, C, H, W = x.shape
    xi = _to_int16_255(x)

    # p_apply gating
    p_apply = float(recipe.p_apply)
    if p_apply < 1.0:
        u = _stateless_uniform(seeds, (), key=10_001, device=device).view(B)
        do = (u < p_apply).view(B, 1, 1, 1)
    else:
        do = torch.ones((B, 1, 1, 1), device=device, dtype=torch.bool)

    hcell, wcell = recipe.S
    beta = float(recipe.beta)
    alpha = float(recipe.alpha)

    def sample_offsets(key_base: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if recipe.offset is None or recipe.offset == "random" or recipe.offset == "random_each_iter":
            oy_u = _stateless_uniform(seeds, (), key=key_base + 1, device=device).view(B)
            ox_u = _stateless_uniform(seeds, (), key=key_base + 2, device=device).view(B)
            oy = torch.floor(oy_u * max(1, int(hcell))).to(torch.int64)
            ox = torch.floor(ox_u * max(1, int(wcell))).to(torch.int64)
            return oy, ox
        if isinstance(recipe.offset, tuple) and len(recipe.offset) == 2:
            oy0, ox0 = recipe.offset
            oy = torch.full((B,), int(oy0) % max(1, int(hcell)), device=device, dtype=torch.int64)
            ox = torch.full((B,), int(ox0) % max(1, int(wcell)), device=device, dtype=torch.int64)
            return oy, ox
        raise ValueError(f"Unsupported offset spec: {recipe.offset}")

    if recipe.offset != "random_each_iter":
        oy, ox = sample_offsets(20_000)

    for it in range(int(recipe.N)):
        if recipe.offset == "random_each_iter":
            oy, ox = sample_offsets(20_000 + 1000 * it)

        allowed_hw = _allowed_mask_horiz(B, H, W, int(wcell), ox, alpha, seeds, key=30_000 + it, device=device)
        allowed_h1w = _allowed_mask_vert(B, H, W, int(hcell), oy, alpha, seeds, key=40_000 + it, device=device)

        xi = _pass_right_batched(xi, beta, allowed_hw,  seeds, key=100_000 + it * 10 + 0)
        xi = _pass_left_batched( xi, beta, allowed_hw,  seeds, key=100_000 + it * 10 + 1)
        xi = _pass_down_batched( xi, beta, allowed_h1w, seeds, key=100_000 + it * 10 + 2)
        xi = _pass_up_batched(   xi, beta, allowed_h1w, seeds, key=100_000 + it * 10 + 3)

    xi = xi.clamp(0, 255)
    out = _to_float01(xi)

    if p_apply < 1.0:
        x0 = x if x.dtype.is_floating_point else _to_float01(_to_int16_255(x))
        out = torch.where(do, out, x0)

    return out


# -----------------------------
# Dataset wrapper: (K+1)x expansion, dynamic/fixed
# -----------------------------
class ShakeAugDataset(Dataset):
    """
    Wrap a base dataset (x,y) and expand it into (K+1) copies:
      k=0 -> original
      k=1..K -> recipe[k-1] augmented copy

    dynamic:
      - True  : augmentation changes with epoch (seed includes epoch)
      - False : augmentation fixed per (i,k) forever

    apply_in_getitem:
      - True  : do augmentation per-sample inside __getitem__
      - False : return meta (x,y,ridx,seed); use ShakeAugBatchCollator for batch-vectorized augmentation
    """
    def __init__(
        self,
        base: Dataset,
        recipes: List[ShakeRecipe],
        *,
        include_original: bool = True,
        dynamic: bool = True,
        base_seed: int = 0,
        apply_in_getitem: bool = False,
    ):
        self.base = base
        self.recipes = list(recipes)
        self.include_original = include_original
        self.dynamic = dynamic
        self.base_seed = int(base_seed)
        self.apply_in_getitem = apply_in_getitem
        self._epoch = 0

        if not include_original and len(self.recipes) == 0:
            raise ValueError("No recipes provided and include_original=False -> empty dataset.")

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __len__(self) -> int:
        n = len(self.base)
        k = len(self.recipes) + (1 if self.include_original else 0)
        return n * k

    def _index_map(self, idx: int) -> Tuple[int, int]:
        n = len(self.base)
        block = idx // n
        i = idx % n
        if self.include_original:
            ridx = -1 if block == 0 else (block - 1)  # -1 => original
        else:
            ridx = block
        return i, ridx

    def _make_seed(self, i: int, ridx: int) -> int:
        seed = self.base_seed
        seed += i * 1_000_003
        seed += (ridx + 2) * 100_07  # +2 to avoid ridx=-1 corner
        if self.dynamic:
            seed += self._epoch * 10_000_019
        return int(seed)

    def __getitem__(self, idx: int):
        i, ridx = self._index_map(idx)
        x, y = self.base[i]
        seed = self._make_seed(i, ridx)

        if not self.apply_in_getitem:
            return x, y, ridx, seed

        if ridx < 0:
            return x, y

        x1 = x.unsqueeze(0)
        out = apply_shake_recipe_batch(
            x1,
            self.recipes[ridx],
            seeds=torch.tensor([seed], dtype=torch.int64, device=x.device),
        )
        return out[0], y


# -----------------------------
# Collate: batch-vectorized augmentation grouped by recipe id
# -----------------------------
class ShakeAugBatchCollator:
    """
    Collate function for ShakeAugDataset(apply_in_getitem=False).
    Groups samples by recipe id (ridx) and applies batch-vectorized shake per group.
    """
    def __init__(self, recipes: List[ShakeRecipe]):
        self.recipes = list(recipes)

    def __call__(self, batch):
        xs, ys, ridxs, seeds = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.as_tensor(ys)
        ridx = torch.as_tensor(ridxs, dtype=torch.int64)
        seed = torch.as_tensor(seeds, dtype=torch.int64)

        x_out = x if x.dtype.is_floating_point else _to_float01(_to_int16_255(x))

        for k in torch.unique(ridx):
            kk = int(k.item())
            if kk < 0:
                continue
            mask = (ridx == kk)
            if mask.any():
                xb = x_out[mask]
                sb = seed[mask]
                x_out[mask] = apply_shake_recipe_batch(xb, self.recipes[kk], seeds=sb)

        return x_out, y
