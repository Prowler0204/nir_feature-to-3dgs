"""
Precompute frozen DINOv2 (register) patch features for NIR images, reduce
384D -> 32D with a PCA fit once over all images, and save per-image targets.

Outputs (in --out_dir):
  <stem>.pt          fp16 tensor [gh, gw, 32]
  pca.npz            mean [384], components [32, 384], explained_variance_ratio
  vis/<stem>.png     first-3-PCA-component visualization (sanity check)

Usage:
  python precompute_vit.py --in_dir data/nir_images --out_dir data/nir_features_32d
"""
import argparse
import glob
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

PATCH = 14


def grid_size(w, h, long_side_patches):
    """Keep aspect ratio; long side = long_side_patches patches (1920x1080, 66 -> 66x37)."""
    if w >= h:
        gw = long_side_patches
        gh = max(1, round(h / w * gw))
    else:
        gh = long_side_patches
        gw = max(1, round(w / h * gh))
    return gw, gh


def load_nir(path):
    """Load NIR as 3-channel uint8 PIL image. Handles 8-bit and 16-bit/float files."""
    img = Image.open(path)
    if img.mode.startswith("I") or img.mode == "F":
        a = np.asarray(img).astype(np.float32)
        lo, hi = np.percentile(a, (1, 99))
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        img = Image.fromarray((a * 255).astype(np.uint8))
    return img.convert("L").convert("RGB")  # replicate single channel to 3


def list_images(d):
    exts = ("jpg", "jpeg", "png", "tif", "tiff")
    paths = []
    for e in exts:
        paths += glob.glob(os.path.join(d, f"*.{e}")) + glob.glob(os.path.join(d, f"*.{e.upper()}"))
    return sorted(set(paths))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/nir_images/")
    ap.add_argument("--out_dir", default="data/nir_features_32d/")
    ap.add_argument("--model", default="dinov2_vits14_reg")
    ap.add_argument("--long_side", type=int, default=66, help="patches along the long image side")
    ap.add_argument("--samples_per_img", type=int, default=2000, help="patches sampled per image for PCA fit")
    ap.add_argument("--n_components", type=int, default=32)
    ap.add_argument("--n_vis", type=int, default=8, help="number of PCA visualizations to dump")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "vis"), exist_ok=True)

    paths = list_images(args.in_dir)
    if not paths:
        print(f"No images found in {args.in_dir}")
        return

    # LOAD FROM LOCAL REPO TO AVOID PYTHON 3.10 TYPE HINT CRASHES
    model = torch.hub.load("D:/Chirayu/dinov2", args.model, source="local").to(device).eval()
    
    norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # ---- Pass 1: extract 384D patch tokens (kept in fp16 on CPU) ----
    feats, grids = [], []
    for p in tqdm(paths, desc="Extracting 384D features"):
        img = load_nir(p)
        gw, gh = grid_size(*img.size, args.long_side)
        img = img.resize((gw * PATCH, gh * PATCH), Image.Resampling.BICUBIC)
        x = norm(img).unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            out = model.forward_features(x)
        tok = out["x_norm_patchtokens"][0]  # [gh*gw, 384]
        assert tok.shape[0] == gh * gw, (tok.shape, gh, gw)
        feats.append(tok.half().cpu())
        grids.append((gh, gw))

    # ---- Fit PCA on a random subsample of patches from every image ----
    rng = np.random.default_rng(0)
    sample = []
    for f in feats:
        idx = rng.choice(f.shape[0], size=min(args.samples_per_img, f.shape[0]), replace=False)
        sample.append(f[idx].float())
    sample = torch.cat(sample).numpy()
    print(f"Fitting PCA {sample.shape[1]}D -> {args.n_components}D on {sample.shape[0]} patches...")
    pca = PCA(n_components=args.n_components, random_state=0).fit(sample)
    evr = pca.explained_variance_ratio_
    print(f"Explained variance retained: {evr.sum():.3f}")

    mean = torch.from_numpy(pca.mean_).float()
    comps = torch.from_numpy(pca.components_).float()  # [32, 384]
    np.savez(
        os.path.join(args.out_dir, "pca.npz"),
        mean=pca.mean_, components=pca.components_, explained_variance_ratio=evr,
    )

    # ---- Pass 2: project, save targets, dump visualizations ----
    # Global percentiles of first 3 comps so visualizations are comparable across views
    proj_all = [((f.float() - mean) @ comps.T) for f in feats]
    vis_lo = torch.stack([p[:, :3].quantile(0.02, dim=0) for p in proj_all]).mean(0)
    vis_hi = torch.stack([p[:, :3].quantile(0.98, dim=0) for p in proj_all]).mean(0)

    for i, (p, proj) in enumerate(tqdm(list(zip(paths, proj_all)), desc="Saving targets")):
        gh, gw = grids[i]
        stem = os.path.splitext(os.path.basename(p))[0]
        torch.save(proj.reshape(gh, gw, -1).half(), os.path.join(args.out_dir, f"{stem}.pt"))

        if i < args.n_vis:
            v = ((proj[:, :3] - vis_lo) / (vis_hi - vis_lo + 1e-6)).clamp(0, 1)
            v = (v.reshape(gh, gw, 3).numpy() * 255).astype(np.uint8)
            Image.fromarray(v).resize((gw * 8, gh * 8), Image.Resampling.NEAREST).save(
                os.path.join(args.out_dir, "vis", f"{stem}.png"))

    print(f"Done. {len(paths)} targets + pca.npz saved to {args.out_dir}")


if __name__ == "__main__":
    main()
