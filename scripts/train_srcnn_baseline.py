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
    train_hr_dir: str = "data/train_HR"
    val_hr_dir: str = "data/valid_HR"

    scale: int = 4                 # adaugat fata de 2
    patch_size_hr: int = 128       #  adaugat fata de 2 (la x4 LR patch = 32 px → mai mult context, mai bine)
    batch_size: int = 64           # adaugat fata de 2, speram ca duce serverul
    epochs: int = 200              # adaugat fata de 2
    lr: float = 1e-4
    num_workers: int = 2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "srcnn_x4.pth"
    seed: int = 42

cfg = Config()

torch.manual_seed(cfg.seed)

# -------------------------
# Utils
# -------------------------
def charbonnier_loss(x, y, eps=1e-3): # adaugat fata de 3
    return torch.mean(torch.sqrt((x - y) ** 2 + eps ** 2))

def list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return sorted(paths)

def pil_rgb_to_y_tensor(img_rgb: Image.Image) -> torch.Tensor:
    """
    img_rgb: PIL RGB
    return: Y channel ca torch Tensor (1,H,W) in [0,1]
    """
    ycbcr = img_rgb.convert("YCbCr")
    y, _, _ = ycbcr.split()
    return transforms.ToTensor()(y)

def calc_psnr(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-10) -> float:
    """
    sr, hr: tensors (1,H,W) in [0,1]
    """
    mse = torch.mean((sr - hr) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)

def ssim_simple(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    SSIM simplificat (global), suficient pt proiect (nu e implementarea oficială multi-scale).
    sr/hr: (1,H,W)
    """
    # flatten
    x = sr.reshape(-1)
    y = hr.reshape(-1)

    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var(unbiased=False)
    sigma_y = y.var(unbiased=False)
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return float(ssim.item())

# -------------------------
# Dataset: HR -> (bicubic downscale) -> LR, apoi upsample LR la HR (bicubic)
# SRCNN învață: bicubic-upsampled -> HR
# Clasic: se lucrează pe canalul Y (luminanță), ca în lucrarea SRCNN originală
# -------------------------
class SRDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int, patch_size_hr: int, training: bool):
        self.paths = list_images(hr_dir)
        if len(self.paths) == 0:
            raise RuntimeError(f"Nu am găsit imagini în: {hr_dir}")

        self.scale = scale
        self.patch_size_hr = patch_size_hr
        self.training = training
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def _random_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ps = self.patch_size_hr
        # asigură multiplu de scale (ca LR să fie întreg)
        ps = (ps // self.scale) * self.scale
        if w < ps or h < ps:
            # dacă e mică, o mărim puțin (rar pe DIV2K)
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

        # crop
        hr_img = self._random_crop(hr_img) if self.training else self._center_crop(hr_img)

        # -------------------------
        # Data augmentation (train only)
        # -------------------------
        if self.training:
            # flip orizontal
            if torch.rand(1).item() < 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)

            # flip vertical
            if torch.rand(1).item() < 0.5:
                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)

            # rotație 0 / 90 / 180 / 270
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                hr_img = hr_img.rotate(90 * k)


        # LR: downscale
        w, h = hr_img.size
        lr_img = hr_img.resize((w // self.scale, h // self.scale), Image.BICUBIC)
        # bicubic upsample la dim HR (input SRCNN)
        bicubic_img = lr_img.resize((w, h), Image.BICUBIC)

        # folosim YCbCr -> adaugat fata de 1
        hr_y = pil_rgb_to_y_tensor(hr_img)      # (1,H,W)
        bic_y = pil_rgb_to_y_tensor(bicubic_img)    # (1,H,W)       

        return bic_y, hr_y

# -------------------------
# SRCNN model (3 conv layers)
# Original paper: 9x9, 1x1, 5x5; num filters 64, 32
# -------------------------
class SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

        # init decent
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
    running = 0.0
    for bic, hr in loader:
        bic = bic.to(device)
        hr = hr.to(device)

# adaugat fata de 3
        res = model(bic)
        sr = bic + res
        loss = charbonnier_loss(sr, hr) # adaugat fata de 3

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        running += loss.item() * bic.size(0)

    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    psnr_vals = []
    ssim_vals = []
    loss_vals = []

    for bic, hr in loader:
        bic = bic.to(device)
        hr = hr.to(device)

# adaugat fata de 3
        res = model(bic)
        sr = bic + res
        loss = charbonnier_loss(sr, hr) # adaugat fata de 3
        loss_vals.append(loss.item())

        # clamp in [0,1] pentru metrici
        sr_c = torch.clamp(sr, 0.0, 1.0)

        # calc metrics per-sample WITH border crop (standard SR)
        b = cfg.scale
        for i in range(sr_c.size(0)):
            sr_i = sr_c[i]
            hr_i = hr[i]

            # border crop
            if sr_i.size(-1) > 2 * b and sr_i.size(-2) > 2 * b:
                sr_i = sr_i[:, b:-b, b:-b]
                hr_i = hr_i[:, b:-b, b:-b]

            psnr_vals.append(calc_psnr(sr_i, hr_i))
            ssim_vals.append(ssim_simple(sr_i, hr_i))

    return float(sum(loss_vals) / len(loss_vals)), float(sum(psnr_vals) / len(psnr_vals)), float(sum(ssim_vals) / len(ssim_vals))

def main():
    print("Device:", cfg.device)
    train_ds = SRDataset(cfg.train_hr_dir, cfg.scale, cfg.patch_size_hr, training=True)
    val_ds   = SRDataset(cfg.val_hr_dir,   cfg.scale, cfg.patch_size_hr, training=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = SRCNN().to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( # adaugat fata de 3
    optim,
    T_max=cfg.epochs
    )

    best_psnr = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, cfg.device)
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, cfg.device)
        scheduler.step() # adaugat fata de 1
        current_lr = optim.param_groups[0]["lr"] # learning rate

        print(f"[{epoch:03d}/{cfg.epochs}] LR={current_lr:.2e} "
              f"train_loss={tr_loss:.6f}  val_loss={val_loss:.6f}  PSNR={val_psnr:.2f}dB  SSIM={val_ssim:.4f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {"model": model.state_dict(), "scale": cfg.scale},
                cfg.save_path
            )
            print(f"  ✅ Saved best checkpoint: {cfg.save_path} (PSNR {best_psnr:.2f}dB)")

if __name__ == "__main__":
    main()
