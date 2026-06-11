import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    train_hr_dir: str = "data/coco_da_train/images"
    val_hr_dir: str = "data/coco_da_val/images"

    train_label_dir: str = "data/coco_da_train/labels"
    val_label_dir: str = "data/coco_da_val/labels"

    scale: int = 4
    patch_size_hr: int = 128
    batch_size: int = 32
    epochs: int = 120
    lr: float = 1e-5
    num_workers: int = 2

    grad_low: float = 0.10
    grad_high: float = 0.35

    lambda_smooth: float = 1.0
    lambda_texture: float = 1.0
    lambda_edge: float = 1.0

    beta_object: float = 1.5

    eps_charb: float = 1e-3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_path: str = "srcnn_x4_adaptive_3region_thr1035.pth"
    save_path: str = "srcnn_x4_detection_aware_thr1035_beta15.pth"

    seed: int = 42

cfg = Config()
torch.manual_seed(cfg.seed)

# -------------------------
# Utils
# -------------------------
def build_object_mask_from_yolo(label_path: str, img_w: int, img_h: int) -> torch.Tensor:
    """
    YOLO label format:
    class x_center y_center width height
    toate normalizate între 0 și 1
    """
    mask = torch.zeros((1, img_h, img_w), dtype=torch.float32)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        _, xc, yc, bw, bh = parts

        xc = float(xc) * img_w
        yc = float(yc) * img_h
        bw = float(bw) * img_w
        bh = float(bh) * img_h

        x1 = int(max(0, xc - bw / 2))
        y1 = int(max(0, yc - bh / 2))
        x2 = int(min(img_w, xc + bw / 2))
        y2 = int(min(img_h, yc + bh / 2))

        mask[:, y1:y2, x1:x2] = 1.0

    return mask

def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def pil_rgb_to_y_tensor(img_rgb: Image.Image) -> torch.Tensor:
    ycbcr = img_rgb.convert("YCbCr")
    y, _, _ = ycbcr.split()
    return transforms.ToTensor()(y)

def calc_psnr(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-10) -> float:
    mse = torch.mean((sr - hr) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def ssim_simple(sr: torch.Tensor, hr: torch.Tensor) -> float:
    x = sr.reshape(-1)
    y = hr.reshape(-1)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    )
    return float(ssim.item())

def charbonnier_per_pixel(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((x - y) ** 2 + eps ** 2)

def charbonnier_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return charbonnier_per_pixel(x, y, eps=eps).mean()

# -------------------------
# Region masks from Sobel
# -------------------------
def sobel_gradient_map(img: torch.Tensor) -> torch.Tensor:
    """
    img: (B,1,H,W) in [0,1]
    return: normalized gradient magnitude map (B,1,H,W) in [0,1]
    """
    device = img.device
    dtype = img.dtype

    sobel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=dtype, device=device
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=dtype, device=device
    ).unsqueeze(0)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)

    grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)

    B = grad.size(0)
    grad_flat = grad.view(B, -1)
    g_min = grad_flat.min(dim=1)[0].view(B, 1, 1, 1)
    g_max = grad_flat.max(dim=1)[0].view(B, 1, 1, 1)

    grad_norm = (grad - g_min) / (g_max - g_min + 1e-8)
    return grad_norm

def build_region_masks(hr: torch.Tensor,
                       low_thr: float,
                       high_thr: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    returns:
      grad_norm, smooth_mask, texture_mask, edge_mask
    """
    with torch.no_grad():
        grad_norm = sobel_gradient_map(hr)

        smooth_mask = (grad_norm < low_thr).float()
        texture_mask = ((grad_norm >= low_thr) & (grad_norm < high_thr)).float()
        edge_mask = (grad_norm >= high_thr).float()

    return grad_norm, smooth_mask, texture_mask, edge_mask

def masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if denom.item() < 1:
        # dacă regiunea nu există în batch, întoarce 0 pe device corect
        return torch.zeros((), device=loss_map.device, dtype=loss_map.dtype)
    return (loss_map * mask).sum() / denom

# -------------------------
# 3-region adaptive loss
# -------------------------
def adaptive_3region_loss(sr: torch.Tensor, hr: torch.Tensor, object_mask: torch.Tensor = None):
    """
    smooth  -> MSE
    texture -> Charbonnier
    edge    -> L1

    Detection-aware:
    crește penalizarea în regiunile unde există obiecte.
    """
    grad_norm, smooth_mask, texture_mask, edge_mask = build_region_masks(
        hr, cfg.grad_low, cfg.grad_high
    )

    mse_map = (sr - hr) ** 2
    charb_map = charbonnier_per_pixel(sr, hr, eps=cfg.eps_charb)
    l1_map = torch.abs(sr - hr)

    if object_mask is not None:
        object_mask = object_mask.to(sr.device)

        if object_mask.shape[-2:] != sr.shape[-2:]:
            object_mask = F.interpolate(
                object_mask,
                size=sr.shape[-2:],
                mode="nearest"
            )

        object_weight = 1.0 + cfg.beta_object * object_mask

        mse_map = mse_map * object_weight
        charb_map = charb_map * object_weight
        l1_map = l1_map * object_weight

    loss_smooth = masked_mean(mse_map, smooth_mask)
    loss_texture = masked_mean(charb_map, texture_mask)
    loss_edge = masked_mean(l1_map, edge_mask)

    total = (
        cfg.lambda_smooth * loss_smooth +
        cfg.lambda_texture * loss_texture +
        cfg.lambda_edge * loss_edge
    )

    return total, loss_smooth, loss_texture, loss_edge, grad_norm, smooth_mask, texture_mask, edge_mask

def edge_mae(sr: torch.Tensor, hr: torch.Tensor, threshold: float = 0.45) -> float:
    sr_b = sr.unsqueeze(0)
    hr_b = hr.unsqueeze(0)

    grad = sobel_gradient_map(hr_b)
    mask = (grad >= threshold).float()

    denom = mask.sum().item()
    if denom < 1:
        return 0.0

    mae = (torch.abs(sr_b - hr_b) * mask).sum().item() / denom
    return mae

# -------------------------
# Dataset
# -------------------------
class SRDataset(Dataset):
    def __init__(self, hr_dir: str, label_dir: str, scale: int, patch_size_hr: int, training: bool):
        self.paths = list_images(hr_dir)
        if len(self.paths) == 0:
            raise RuntimeError(f"Nu am găsit imagini în: {hr_dir}")

        self.label_dir = label_dir
        self.scale = scale
        self.patch_size_hr = patch_size_hr
        self.training = training

    def __len__(self):
        return len(self.paths)

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ps = (self.patch_size_hr // self.scale) * self.scale
        if w < ps or h < ps:
            img = img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            w, h = img.size

        x = torch.randint(0, w - ps + 1, (1,)).item()
        y = torch.randint(0, h - ps + 1, (1,)).item()
        return img.crop((x, y, x + ps, y + ps))

    def _center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ps = (self.patch_size_hr // self.scale) * self.scale
        ps = min(ps, w - (w % self.scale), h - (h % self.scale))
        x = (w - ps) // 2
        y = (h - ps) // 2
        return img.crop((x, y, x + ps, y + ps))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_img = Image.open(self.paths[idx]).convert("RGB")
        hr_img = self._center_crop(hr_img)

        w, h = hr_img.size
        lr_img = hr_img.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        bicubic_img = lr_img.resize((w, h), Image.BICUBIC)

        hr_y = pil_rgb_to_y_tensor(hr_img)
        bic_y = pil_rgb_to_y_tensor(bicubic_img)

        # object mask din label YOLO
        img_path = self.paths[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, name + ".txt")

        object_mask = build_object_mask_from_yolo(label_path, w, h)

        return bic_y, hr_y, object_mask


# -------------------------
# SRCNN model (same V4 backbone)
# -------------------------
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, loader, optim, device):
    model.train()
    running_total = 0.0
    running_smooth = 0.0
    running_texture = 0.0
    running_edge = 0.0

    for bic, hr, object_mask in loader:
        bic = bic.to(device)
        hr = hr.to(device)
        object_mask = object_mask.to(device)

        res = model(bic)
        sr = bic + res

        total, loss_smooth, loss_texture, loss_edge, *_ = adaptive_3region_loss(sr, hr, object_mask)

        optim.zero_grad(set_to_none=True)
        total.backward()
        optim.step()

        bs = bic.size(0)
        running_total += total.item() * bs
        running_smooth += loss_smooth.item() * bs
        running_texture += loss_texture.item() * bs
        running_edge += loss_edge.item() * bs

    n = len(loader.dataset)
    return (
        running_total / n,
        running_smooth / n,
        running_texture / n,
        running_edge / n
    )

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_vals = []
    smooth_vals = []
    texture_vals = []
    edge_vals_loss = []

    psnr_vals = []
    ssim_vals = []
    edge_metric_vals = []

    for bic, hr, object_mask in loader:
        bic = bic.to(device)
        hr = hr.to(device)
        object_mask = object_mask.to(device)

        res = model(bic)
        sr = bic + res

        total, loss_smooth, loss_texture, loss_edge, *_ = adaptive_3region_loss(sr, hr, object_mask)

        total_vals.append(total.item())
        smooth_vals.append(loss_smooth.item())
        texture_vals.append(loss_texture.item())
        edge_vals_loss.append(loss_edge.item())

        sr_c = torch.clamp(sr, 0.0, 1.0)

        b = cfg.scale
        for i in range(sr_c.size(0)):
            sr_i = sr_c[i]
            hr_i = hr[i]

            if sr_i.size(-1) > 2 * b and sr_i.size(-2) > 2 * b:
                sr_i = sr_i[:, b:-b, b:-b]
                hr_i = hr_i[:, b:-b, b:-b]

            psnr_vals.append(calc_psnr(sr_i, hr_i))
            ssim_vals.append(ssim_simple(sr_i, hr_i))
            edge_metric_vals.append(edge_mae(sr_i, hr_i, threshold=cfg.grad_high))

    return (
        float(sum(total_vals) / len(total_vals)),
        float(sum(smooth_vals) / len(smooth_vals)),
        float(sum(texture_vals) / len(texture_vals)),
        float(sum(edge_vals_loss) / len(edge_vals_loss)),
        float(sum(psnr_vals) / len(psnr_vals)),
        float(sum(ssim_vals) / len(ssim_vals)),
        float(sum(edge_metric_vals) / len(edge_metric_vals))
    )

def main():
    print("Device:", cfg.device)
    print(
        f"grad_low={cfg.grad_low}, grad_high={cfg.grad_high}, "
        f"lambda_smooth={cfg.lambda_smooth}, "
        f"lambda_texture={cfg.lambda_texture}, "
        f"lambda_edge={cfg.lambda_edge}"
    )

    train_ds = SRDataset(
        cfg.train_hr_dir,
        cfg.train_label_dir,
        cfg.scale,
        cfg.patch_size_hr,
        training=True
    )

    val_ds = SRDataset(
        cfg.val_hr_dir,
        cfg.val_label_dir,
        cfg.scale,
        cfg.patch_size_hr,
        training=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    model = SRCNN().to(cfg.device)

    if os.path.exists(cfg.pretrained_path):
        checkpoint = torch.load(cfg.pretrained_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded pretrained checkpoint: {cfg.pretrained_path}")
    else:
        print(f"WARNING: pretrained checkpoint not found: {cfg.pretrained_path}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=cfg.epochs
    )

    best_psnr = -1.0

    for epoch in range(1, cfg.epochs + 1):
        tr_total, tr_smooth, tr_texture, tr_edge = train_one_epoch(model, train_loader, optim, cfg.device)
        val_total, val_smooth, val_texture, val_edge_loss, val_psnr, val_ssim, val_edge_mae = evaluate(model, val_loader, cfg.device)

        scheduler.step()
        current_lr = optim.param_groups[0]["lr"]

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"LR={current_lr:.2e} "
            f"train_total={tr_total:.6f} "
            f"train_smooth={tr_smooth:.6f} "
            f"train_texture={tr_texture:.6f} "
            f"train_edge={tr_edge:.6f} "
            f"val_total={val_total:.6f} "
            f"val_smooth={val_smooth:.6f} "
            f"val_texture={val_texture:.6f} "
            f"val_edge={val_edge_loss:.6f} "
            f"PSNR={val_psnr:.2f}dB "
            f"SSIM={val_ssim:.4f} "
            f"EdgeMAE={val_edge_mae:.6f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {
                    "model": model.state_dict(),
                    "scale": cfg.scale,
                    "grad_low": cfg.grad_low,
                    "grad_high": cfg.grad_high,
                    "lambda_smooth": cfg.lambda_smooth,
                    "lambda_texture": cfg.lambda_texture,
                    "lambda_edge": cfg.lambda_edge,
                    "beta_object": cfg.beta_object,
                    "pretrained_path": cfg.pretrained_path,
                    "best_psnr": best_psnr,
                    "val_ssim": val_ssim,
                    "val_edge_mae": val_edge_mae,
                    "epoch": epoch
                },
                cfg.save_path
            )
            print(
                f"  Saved best checkpoint: {cfg.save_path} "
                f"(PSNR {best_psnr:.2f}dB, SSIM {val_ssim:.4f}, EdgeMAE {val_edge_mae:.6f})"
            )

if __name__ == "__main__":
    main()