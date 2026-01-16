# test_shakeaug_mnist.py
import os
import argparse
import torch
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from shake_aug import (
    ShakeRecipe,
    apply_shake_recipe_batch,
    ShakeAugDataset,
)

def to_int255(x: torch.Tensor) -> torch.Tensor:
    if x.dtype.is_floating_point:
        return (x * 255.0).round().clamp(0, 255).to(torch.int64)
    if x.dtype == torch.uint8:
        return x.to(torch.int64)
    return x.to(torch.int64).clamp(0, 255)

def assert_sum_conserved(x0_int: torch.Tensor, x1_int: torch.Tensor, name: str):
    s0 = x0_int.sum(dim=(1, 2, 3))
    s1 = x1_int.sum(dim=(1, 2, 3))
    md = (s1 - s0).abs().max().item()
    print(f"[check] {name}: max|sum_after-sum_before| = {md}")
    assert md == 0, f"{name}: sum not conserved! max diff={md}"

def assert_range_ok(x_int: torch.Tensor, name: str):
    mn = x_int.min().item()
    mx = x_int.max().item()
    print(f"[check] {name}: range [{mn},{mx}]")
    assert mn >= 0 and mx <= 255, f"{name}: out-of-range values [{mn},{mx}]"

def cell_sums_aligned(x_int: torch.Tensor, cell_h: int, cell_w: int) -> torch.Tensor:
    B, C, H, W = x_int.shape
    assert H % cell_h == 0 and W % cell_w == 0
    x = x_int.view(B, C, H // cell_h, cell_h, W // cell_w, cell_w)
    return x.sum(dim=(3, 5))

def assert_cell_sum_conserved_aligned(x0_int, x1_int, cell_h, cell_w, name: str):
    cs0 = cell_sums_aligned(x0_int, cell_h, cell_w)
    cs1 = cell_sums_aligned(x1_int, cell_h, cell_w)
    md = (cs1 - cs0).abs().max().item()
    print(f"[check] {name}: max|cell_sum_after-cell_sum_before| = {md}")
    assert md == 0, f"{name}: cell sums changed (aligned grid)! max diff={md}"

def assert_batch_order_invariant(x: torch.Tensor, recipe: ShakeRecipe, seeds: torch.Tensor, device: torch.device):
    x = x.to(device)
    seeds = seeds.to(device)

    out1 = apply_shake_recipe_batch(x, recipe, seeds=seeds)

    perm = torch.randperm(x.shape[0], device=device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(x.shape[0], device=device)

    out2p = apply_shake_recipe_batch(x[perm], recipe, seeds=seeds[perm])
    out2 = out2p[inv]

    o1 = to_int255(out1.cpu())
    o2 = to_int255(out2.cpu())
    md = (o1 - o2).abs().max().item()
    print(f"[check] batch-order invariance: max diff = {md}")
    assert md == 0, f"batch-order invariance failed! max diff={md}"

def save_compare_grid(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, path: str, nrow: int):
    grid = make_grid(torch.cat([x0, x1, x2], dim=0), nrow=nrow, padding=2)
    save_image(grid, path)
    print(f"[save] {path}  (row1=orig,row2=recipe1,row3=recipe2)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--outdir", type=str, default="./runs_shakeaug_test")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dynamic", action="store_true", help="if set, epoch changes will change augmentation")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[info] device = {device}")

    tfm = transforms.ToTensor()
    mnist = datasets.MNIST(root="/home/yansheng/shake_DataEnhancement/data", train=True, transform=tfm)

    B = min(args.B, len(mnist))
    xs = torch.stack([mnist[i][0] for i in range(B)], dim=0)  # (B,1,28,28)
    ys = torch.tensor([mnist[i][1] for i in range(B)], dtype=torch.long)
    print(f"[info] loaded base batch: xs={tuple(xs.shape)} labels[:8]={ys[:8].tolist()}")

    seeds_base = (torch.arange(B, dtype=torch.int64) * 1009 + args.seed).clone()

    x0_int = to_int255(xs)
    assert_range_ok(x0_int, "input")

    # --- invariant tests that should ALWAYS hold ---
    recipe_alpha1 = ShakeRecipe(N=10, beta=0.50, alpha=1.0, S=(4, 4), offset="random_each_iter", p_apply=1.0)
    out_a = apply_shake_recipe_batch(xs.to(device), recipe_alpha1, seeds=seeds_base.to(device)).cpu()
    xA_int = to_int255(out_a)
    assert_range_ok(xA_int, "alpha=1 output")
    assert_sum_conserved(x0_int, xA_int, "alpha=1 global sum")

    recipe_alpha0_fixed = ShakeRecipe(N=10, beta=0.50, alpha=0.0, S=(4, 4), offset=(0, 0), p_apply=1.0)
    out_b = apply_shake_recipe_batch(xs.to(device), recipe_alpha0_fixed, seeds=seeds_base.to(device)).cpu()
    xB_int = to_int255(out_b)
    assert_range_ok(xB_int, "alpha=0 (fixed offset) output")
    assert_sum_conserved(x0_int, xB_int, "alpha=0 (fixed offset) global sum")
    assert_cell_sum_conserved_aligned(x0_int, xB_int, 4, 4, "alpha=0 (fixed offset) cell sum 4x4")

    assert_batch_order_invariant(xs, recipe_alpha1, seeds_base, device)

    # --- epoch loop: visualize dynamic vs fixed ---
    recipes_for_vis = [
        ShakeRecipe(N=3, beta=0.25, alpha=0.2, S=(4, 4), offset="random_each_iter", p_apply=1.0),
        ShakeRecipe(N=500, beta=1, alpha=1, S=(2, 4), offset="random_each_iter", p_apply=1.0),
    ]

    # use dataset wrapper so "dynamic" means seed includes epoch
    ds = ShakeAugDataset(
        mnist,
        recipes_for_vis,
        include_original=True,
        dynamic=args.dynamic,
        base_seed=args.seed,
        apply_in_getitem=True,
    )
    n_base = len(mnist)
    Nvis = 8

    print(f"[info] epoch loop: dynamic={args.dynamic}")
    print("[info] saving epochXXX_compare.png (row1=orig,row2=recipe1,row3=recipe2)")

    prev_checksum = None
    for epoch in range(args.epochs):
        ds.set_epoch(epoch)

        # fixed set of base indices for compare
        idxs = list(range(Nvis))
        orig = torch.stack([ds[i + 0 * n_base][0] for i in idxs], dim=0)
        r1   = torch.stack([ds[i + 1 * n_base][0] for i in idxs], dim=0)
        r2   = torch.stack([ds[i + 2 * n_base][0] for i in idxs], dim=0)

        # checksum on recipe1 row (int space)
        chk = to_int255(r1).sum().item()
        print(f"[epoch {epoch:03d}] checksum(recipe1 row) = {chk}")

        if prev_checksum is not None:
            if args.dynamic:
                # should typically change
                if chk == prev_checksum:
                    print("[warn] dynamic=True but checksum did not change (can happen rarely).")
            else:
                # must not change
                assert chk == prev_checksum, "dynamic=False but epoch output changed!"
        prev_checksum = chk

        cpath = os.path.join(args.outdir, f"epoch{epoch:03d}_compare.png")
        save_compare_grid(orig, r1, r2, cpath, nrow=Nvis)

    print("\n[OK] All invariants passed, and epoch behavior matches dynamic flag.")

if __name__ == "__main__":
    main()
