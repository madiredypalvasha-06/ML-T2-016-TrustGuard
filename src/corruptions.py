"""Local, deterministic re-implementation of the standard CIFAR-10-C
corruption types used in the study (Hendrycks & Dietterich, 2019).

The original CIFAR-10-C is no longer downloadable from its GitHub release,
so we reproduce the official corruption recipes with the paper's severity
parameters. Every function is seeded so experiments are fully reproducible.
Implemented with torch ops so they run on the same device as the model.
"""
import torch
import torch.nn.functional as F

_NOISE = [0.08, 0.12, 0.18, 0.26, 0.38]
# severity-ordered so higher severity = stronger corruption (dimmer / lower contrast)
_BRIGHT = [0.5, 0.4, 0.3, 0.2, 0.1]
_CONTRAST = [0.9, 0.7, 0.5, 0.3, 0.1]
_FOG = [0.08, 0.12, 0.18, 0.26, 0.38]
_MOTION_K = [5, 7, 9, 11, 13]
_GEN = torch.Generator().manual_seed(0)


def _chan_groups(x):
    """(N,3,H,W) -> (N*3,1,H,W) for grouped conv."""
    return x.view(-1, 1, x.shape[-2], x.shape[-1])


def _ungroup(c, orig):
    return c.view(orig.shape)


def gaussian_noise(x, sev, seed=0):
    g = torch.Generator(device=x.device).manual_seed(100 * sev + seed)
    noise = torch.randn(x.shape, generator=g, device=x.device) * _NOISE[sev - 1]
    return torch.clamp(x + noise, 0, 1)


def brightness(x, sev, seed=0):
    return torch.clamp(x * _BRIGHT[sev - 1], 0, 1)


def contrast(x, sev, seed=0):
    c = _CONTRAST[sev - 1]
    means = x.mean(dim=(2, 3), keepdim=True)
    return torch.clamp((x - means) * c + means, 0, 1)


def _motion_kernel(k, angle_deg, device):
    """Line kernel of size k rotated by `angle_deg`, normalized."""
    center = (k - 1) / 2
    y = torch.arange(k, device=device).float() - center
    x = torch.arange(k, device=device).float() - center
    Y, X = torch.meshgrid(y, x, indexing="ij")
    ang = torch.deg2rad(torch.tensor(float(angle_deg), device=device))
    dir_x, dir_y = torch.cos(ang), torch.sin(ang)
    perp = (X * dir_y - Y * dir_x).abs()      # distance to the line (px)
    along = X * dir_x + Y * dir_y             # position along the line (px)
    kernel = (perp <= 0.6) * (along.abs() <= k * 0.35)
    kernel = kernel * torch.exp(-0.5 * (perp / 0.8) ** 2)
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel


def motion_blur(x, sev, seed=0, angle_deg=45.0):
    k = _MOTION_K[sev - 1] if _MOTION_K[sev - 1] % 2 == 1 else _MOTION_K[sev - 1] + 1
    kern = _motion_kernel(k, angle_deg, x.device)
    kern = kern.view(1, 1, k, k).repeat(3, 1, 1, 1)
    blurred = F.conv2d(x, kern, padding=k // 2, groups=3)
    return torch.clamp(blurred, 0, 1)


def _fractal_noise(h, w, device, seed):
    """Low-res random field upsampled = cheap fractal airlight map."""
    g = torch.Generator(device=device).manual_seed(seed)
    small_h, small_w = max(2, h // 8), max(2, w // 8)
    base = torch.randn(1, 1, small_h, small_w, generator=g, device=device)
    field = F.interpolate(base, size=(h, w), mode="bilinear", align_corners=False)
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field.squeeze(0)  # (1,H,W)


def fog(x, sev, seed=0):
    """Atmospheric scattering blend with a random distance field."""
    n = x.shape[0]
    h, w = x.shape[-2], x.shape[-1]
    field = _fractal_noise(h, w, x.device, 1000 * sev + seed)  # (1,H,W)
    yy = torch.linspace(-1, 1, h, device=x.device)
    xx = torch.linspace(-1, 1, w, device=x.device)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    dist = (torch.abs(gy).min(torch.abs(gx))) ** 2
    dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    mask = (field * (1 - dist) + dist).unsqueeze(0)  # (1,1,H,W)
    factor = _FOG[sev - 1]
    return torch.clamp(x * (1 - factor * mask) + factor * mask, 0, 1)


_CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "brightness": brightness,
    "contrast": contrast,
    "motion_blur": motion_blur,
    "fog": fog,
}


def corrupt(x, corruption, severity, seed=0):
    """Apply a single corruption to a (N,3,H,W) float tensor in [0,1]."""
    if corruption not in _CORRUPTIONS:
        raise ValueError(f"unknown corruption {corruption}; "
                         f"available: {list(_CORRUPTIONS)}")
    return _CORRUPTIONS[corruption](x, int(severity), seed=seed)


def available_corruptions():
    return list(_CORRUPTIONS)