import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch

def compute_ssim(img1, img2):
    """
    Compute SSIM between two images, adapting window size if needed.
    Accepts NumPy arrays or torch.Tensors of shape (H,W), (H,W,C), or (C,H,W).
    Returns:
        float or None: SSIM value, or None if computation fails.
    """
    # Convert to NumPy
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # Convert CHW to HWC
    if img1.ndim == 3 and img1.shape[0] in (1, 3):
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] in (1, 3):
        img2 = np.transpose(img2, (1, 2, 0))

    # Compute SSIM safely
    try:
        return ssim(
            img1, img2,
            data_range=img2.max() - img2.min(),
            channel_axis=-1
        )
    except ValueError:
        # Window too large; pick the largest odd win_size <= min(height, width, 7)
        h, w = img2.shape[:2]
        max_odd = lambda x: x if x % 2 == 1 else x - 1
        win_size = min(max_odd(h), max_odd(w), 7)
        if win_size < 3:
            return None
        try:
            return ssim(
                img1, img2,
                data_range=img2.max() - img2.min(),
                channel_axis=-1,
                win_size=win_size
            )
        except Exception:
            return None


def compute_psnr(img1, img2):
    """
    Compute PSNR between two images. Returns None if fails.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    if img1.ndim == 3 and img1.shape[0] in (1, 3):
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.ndim == 3 and img2.shape[0] in (1, 3):
        img2 = np.transpose(img2, (1, 2, 0))
    try:
        return psnr(img1, img2, data_range=img2.max() - img2.min())
    except Exception:
        return None


def compute_dice(img1, img2, mask):
    """
    Compute Dice coefficient between two images within a binary mask.
    Returns:
        float: Dice score, or None if mask/image shapes mismatch.
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if mask.shape != img1.shape[:2]:
        return None

    m = mask > 0
    img1m = img1[m]
    img2m = img2[m]

    img1b = (img1m > 0.5).astype(np.float32)
    img2b = (img2m > 0.5).astype(np.float32)

    intersection = np.sum(img1b * img2b)
    denom = np.sum(img1b) + np.sum(img2b)
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom
