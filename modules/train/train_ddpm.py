import torch


import glob
import os
import csv
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from accelerate import Accelerator
from tqdm import tqdm

from core import config
from modules.dataloader.dataloader_ddpm import PairedImageDataset
from core.evaluate import evaluate_all
from core.image_io import load_image_tensor
from modules.refiner.ddpm_refine import DDPMRefiner


def _first_image(dir_path):
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files = glob.glob(os.path.join(dir_path, ext))
        if files:
            return files[0]
    raise RuntimeError(f"No images found in {dir_path}")


def train_ddpm():
    cfg = config.load_config()
    accelerator = Accelerator()

    image_size = cfg['cyclegan']['image_size']
    batch_size = cfg['ddpm']['batch_size']
    epochs = cfg['ddpm']['epochs']
    lr = cfg['ddpm']['lr']

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  
    ])

    dataset = PairedImageDataset(
        image_dir=cfg['paths']['paired_low'],
        label_dir=cfg['paths']['ddpm_paired'],
        transform=transform
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No paired DDPM images found under:\n"
            f"  image_dir = {cfg['paths']['paired_low']}\n"
            f"  label_dir = {cfg['paths']['ddpm_paired']}"
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Try to load model from config, otherwise initialize from scratch
    try:
        model = UNet2DModel.from_pretrained(cfg['paths']['ddpm_model'])
        print("✅ Loaded existing DDPM model from checkpoint.")
    except Exception:
        print("\u26A0\uFE0F No config found. Initializing model from scratch.")
        model = UNet2DModel(
            sample_size=image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
        )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    os.makedirs(cfg['paths']['ddpm_model'], exist_ok=True)
    eval_log = os.path.join(cfg['paths']['ddpm_model'], "ddpm_eval_log.csv")
    with open(eval_log, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "PSNR", "SSIM", "Dice", "SemanticLoss"])

    val_in = _first_image(cfg['paths']['paired_low'])
    val_tgt = _first_image(cfg['paths']['ddpm_paired'])

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"DDPM Epoch {ep}/{epochs}"):
            images = batch['input'].to(accelerator.device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device).long()
            noisy = noise_scheduler.add_noise(images, noise, timesteps)
            pred = model(noisy, timesteps).sample

            loss = torch.nn.functional.mse_loss(pred, noise)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        accelerator.print(f"Epoch {ep} avg loss: {total_loss/len(dataloader):.4f}")

        img_in = load_image_tensor(val_in, device=accelerator.device, force_grayscale=True)
        img_tgt = load_image_tensor(val_tgt, device=accelerator.device, force_grayscale=True)
        with torch.no_grad():
            refiner = DDPMRefiner(model=model, scheduler=noise_scheduler, device=accelerator.device)
            refined = refiner.refine(img_in)

        samp_dir = os.path.join(cfg['paths']['ddpm_model'], "eval_samples")
        os.makedirs(samp_dir, exist_ok=True)
        samp_p = os.path.join(samp_dir, f"epoch_{ep}.png")
        if isinstance(refined, Image.Image):
            refined.save(samp_p)
        else:
            refined_img = refined.squeeze()
            if refined_img.ndim == 3:
                refined_img = refined_img.squeeze(0)  # remove channel dim
            transforms.ToPILImage()(refined_img.cpu()).save(samp_p)


        res = evaluate_all(val_tgt, samp_p, mask_path=None, device=accelerator.device)

        with open(eval_log, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ep, res["psnr"], res["ssim"], res["dice"], res["semantic_loss"]])

        ssim_val = f"{res['ssim']:.4f}" if res['ssim'] is not None else "None"
        dice_val = f"{res['dice']:.4f}" if res['dice'] is not None else "NA"

        accelerator.print(
            f"[EVAL] Ep {ep}: PSNR={res['psnr']:.2f}, SSIM={ssim_val}, Dice={dice_val}, Sem={res['semantic_loss']:.5f}"
        )


        if accelerator.is_main_process:
            model.save_pretrained(cfg['paths']['ddpm_model'])
            UNet2DModel.save_config(model, save_directory=cfg['paths']['ddpm_model'])
            DDPMScheduler.save_config(noise_scheduler, save_directory=cfg['paths']['ddpm_model'])
            noise_scheduler.save_pretrained(cfg['paths']['ddpm_model'])


if __name__ == "__main__":
    train_ddpm()
