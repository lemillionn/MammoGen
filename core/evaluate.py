# core/evaluate.py

import os
import torch
from core.metrics import compute_psnr, compute_ssim, compute_dice
from core.image_io import load_image_tensor, load_mask
from modules.semantics.feature_extractor import FeatureExtractor
from modules.semantics.feature_loss import FeatureLoss
import torch.nn.functional as F
import torchvision.transforms as T


def evaluate_all(
    real_path: str,
    generated_path: str,
    mask_path: str = None,
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    Compute PSNR, SSIM, optional Dice, and DINO‐based semantic loss.
    Ensures both images are grayscale and same size.
    """
    # 1) load into [-1,1] tensors
    img_real = load_image_tensor(real_path, device=device, force_grayscale=True)
    img_fake = load_image_tensor(generated_path, device=device, force_grayscale=True)

    # Resize fake to match real (SSIM requires same shape)
    if img_real.shape != img_fake.shape:
        img_fake = F.interpolate(img_fake, size=img_real.shape[-2:], mode='bilinear', align_corners=False)

    # 2) pixel metrics
    psnr_val = compute_psnr(img_real, img_fake)
    # Ensure same spatial size and channels
    if img_real.shape[-2:] != img_fake.shape[-2:]:
        img_fake = F.interpolate(img_fake, size=img_real.shape[-2:], mode='bilinear', align_corners=False)

    if img_real.shape[1] != img_fake.shape[1]:
        img_fake = img_fake[:, :img_real.shape[1], :, :]  # Trim extra channels if needed

    def unnormalize(tensor):
        return (tensor + 1) / 2  # works for grayscale

    img_real_unnorm = unnormalize(img_real)
    img_fake_unnorm = unnormalize(img_fake)

    try:
        ssim_val = compute_ssim(img_real_unnorm, img_fake_unnorm)
    except Exception as e:
        print(f"⚠️ SSIM computation failed: {e}")
        ssim_val = None

    ssim_str = f"{ssim_val:.4f}" if ssim_val is not None else "None"

    print(f"[DEBUG] Real shape: {img_real.shape}, Fake shape: {img_fake.shape}")

    # 3) optional Dice
    dice_val = None
    if mask_path and os.path.exists(mask_path):
        mask = load_mask(mask_path, device=device)
        dice_val = compute_dice(img_real, img_fake, mask)

    # 4) semantic loss
    extractor = FeatureExtractor(device=device)
    if hasattr(extractor, "eval"):
        extractor.eval()
    loss_fn = FeatureLoss()
    with torch.no_grad():
        feat_real = extractor(img_real)
        feat_fake = extractor(img_fake)
        semantic_loss = float(loss_fn(feat_real, feat_fake))

    # log
    log_msg = (
        f"\n[EVAL] PSNR: {psnr_val:.2f}  SSIM: {ssim_str}"
        + (f"  Dice: {dice_val:.4f}" if dice_val is not None else "")
        + f"  SemanticLoss: {semantic_loss:.6f}\n"
    )
    print(log_msg)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "dice": dice_val,
        "semantic_loss": semantic_loss
    }


if __name__ == "__main__":
    _ = evaluate_all(
        real_path="data/paired/subtracted_cesm/P1_L_DM_MLO.png",
        generated_path="data/outputs/final/synthetic_refined.png",
        mask_path=None,
        device=torch.device("cpu")
    )
