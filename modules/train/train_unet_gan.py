import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core import config
from core.transforms_unet import get_unet_train_transforms as get_tf
from modules.dataloader.dataloader_paired import PairedImageDataset
from modules.translator.generator_unet import UNetGenerator
from modules.translator.discriminator import Discriminator
from modules.semantics.semantic_loss_wrapper import SemanticLossWrapper
from modules.translator.unet_gan_infer import save_unet_gan_samples

# Optional DINOv2
import timm
try:
    from transformers import Dinov2Model
    HF_DINO = True
except ImportError:
    Dinov2Model = None
    HF_DINO = False


def denorm(x):
    """[-1,1] → [0,1] for saving/viewing."""
    return (x.clamp(-1,1) + 1.0) * 0.5


def train_unet_gan():
    cfg    = config.load_config()
    device = torch.device(cfg['unet_gan']['device'])

    # --- Data ---
    tf = get_tf(cfg['unet_gan']['image_size'])
    ds = PairedImageDataset(
        cfg['paths']['paired_low'],
        cfg['paths']['paired_sub'],
        transform=tf
    )
    dl = DataLoader(
        ds,
        batch_size=cfg['unet_gan']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # --- Models ---
    G   = UNetGenerator(in_channels=3, out_channels=3).to(device)
    D   = Discriminator(input_nc=3).to(device)
    sem = SemanticLossWrapper(device)

    # --- Losses & Optimizers ---
    adv   = torch.nn.MSELoss()
    l1    = torch.nn.L1Loss()
    g_opt = torch.optim.Adam(G.parameters(), lr=cfg['unet_gan']['lr'], betas=(0.5,0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg['unet_gan']['lr'], betas=(0.5,0.999))

    λ1       = cfg['unet_gan']['lambda_l1']
    λsem     = cfg['unet_gan']['lambda_semantic']
    λdino    = cfg['unet_gan'].get('lambda_dino', 0.0)

    # --- Load DINOv2 if enabled ---
    dino = None
    if λdino > 0:
        dino_cfg   = cfg['dinov2']
        model_name = dino_cfg['model_name']
        cache_dir  = dino_cfg.get('local_cache', None)
        try:
            dino = timm.create_model(model_name, pretrained=True)
            if hasattr(dino, 'head'):
                dino.head = torch.nn.Identity()
        except Exception:
            if not HF_DINO:
                raise ImportError("Install transformers to load DINOv2 from HF")
            dino = Dinov2Model.from_pretrained(model_name, cache_dir=cache_dir)
        dino.to(device).eval()
        for p in dino.parameters():
            p.requires_grad = False

    # --- Outputs & Metrics ---
    os.makedirs(cfg['unet_gan']['output_samples'], exist_ok=True)
    os.makedirs(cfg['unet_gan']['checkpoints'],   exist_ok=True)
    metrics = []
    best_loss = float('inf')

    # --- Training Loop ---
    for epoch in range(cfg['unet_gan']['epochs']):
        sum_loss_G = 0.0
        sum_loss_D = 0.0
        n_batches  = 0

        print(f"Epoch {epoch+1}/{cfg['unet_gan']['epochs']}")
        for i, (low, sub) in enumerate(tqdm(dl)):
            low, sub = low.to(device), sub.to(device)
            n_batches += 1

            # Generator step
            g_opt.zero_grad()
            fake = G(low)  # outputs in [-1,1]
            loss_adv  = adv(D(fake), torch.ones_like(D(fake)))
            loss_l1   = λ1 * l1(fake, sub)
            loss_sem  = λsem * sem.compute_loss(sub, fake)

            # DINO loss
            loss_dino = 0.0
            if λdino > 0:
                if HF_DINO and isinstance(dino, Dinov2Model):
                    real_feats = dino(pixel_values=sub).last_hidden_state[:,0,:]
                    fake_feats = dino(pixel_values=fake).last_hidden_state[:,0,:]
                else:
                    real_feats = dino(sub)
                    fake_feats = dino(fake)
                loss_dino = λdino * F.mse_loss(fake_feats, real_feats)

            loss_G = loss_adv + loss_l1 + loss_sem + loss_dino
            loss_G.backward()
            g_opt.step()

            # Discriminator step
            d_opt.zero_grad()
            loss_D = 0.5 * (
                adv(D(sub), torch.ones_like(D(sub))) +
                adv(D(fake.detach()), torch.zeros_like(D(fake)))
            )
            loss_D.backward()
            d_opt.step()

            # Save sample images
            if i % cfg['unet_gan']['sample_interval'] == 0:
                save_unet_gan_samples(
                    input_images=denorm(low),
                    output_images=denorm(fake),
                    target_images=denorm(sub),
                    epoch=epoch, batch=i,
                    save_dir=cfg['unet_gan']['output_samples'],
                    prefix_types=('real_low','fake_sub','real_sub')
                )

            sum_loss_G += loss_G.item()
            sum_loss_D += loss_D.item()

        # Compute averages
        avg_G = sum_loss_G / n_batches
        avg_D = sum_loss_D / n_batches
        metrics.append({'epoch': epoch,
                        'loss_G': avg_G,
                        'loss_D': avg_D})

        print(f"  Avg G Loss: {avg_G:.4f} | Avg D Loss: {avg_D:.4f}")

        # Save best model
        if avg_G < best_loss:
            best_loss = avg_G
            torch.save(G.state_dict(),
                       os.path.join(cfg['unet_gan']['checkpoints'], 'G_unet_best.pth'))
            torch.save(D.state_dict(),
                       os.path.join(cfg['unet_gan']['checkpoints'], 'D_unet_best.pth'))

        # Save epoch checkpoint
        torch.save(G.state_dict(),
                   os.path.join(cfg['unet_gan']['checkpoints'], f"G_epoch_{epoch}.pth"))
        torch.save(D.state_dict(),
                   os.path.join(cfg['unet_gan']['checkpoints'], f"D_epoch_{epoch}.pth"))

    # --- Save metrics.csv ---
    with open('unet_gan_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print("✅ UNet GAN training complete.")

if __name__ == "__main__":
    train_unet_gan()
