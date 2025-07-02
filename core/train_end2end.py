import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

from core.config import load_config
from modules.dataloader.dataloader_cyclegan import UnpairedImageDataset
from modules.dataloader.dataloader_paired import PairedImageDataset
from core.transforms import get_train_transforms
from core.transforms_unet import get_unet_train_transforms
from modules.translator.generator import Generator as ResnetGenerator
from modules.translator.generator_unet import UNetGenerator as ResnetGeneratorUNet
from modules.translator.discriminator import Discriminator
from modules.semantics.feature_extractor import FeatureExtractor as DINOFeatureExtractor

class End2EndDataset(Dataset):
    def __init__(self, cfg, tf_stage1, tf_stage2):
        self.ds1 = UnpairedImageDataset(
            cfg['paths']['unpaired_rsna'],
            cfg['paths']['unpaired_low'],
            transform=tf_stage1
        )
        self.ds2 = PairedImageDataset(
            cfg['paths']['paired_low'],
            cfg['paths']['paired_sub'],
            transform=tf_stage2
        )
    def __len__(self):
        return min(len(self.ds1), len(self.ds2))
    def __getitem__(self, idx):
        std, low1 = self.ds1[idx]
        _, sub    = self.ds2[idx]
        return std, low1, sub

def main():
    cfg    = load_config("core/config.yaml")
    device = torch.device(cfg["cyclegan"]["device"])

    # Transforms & DataLoader
    tf1 = get_train_transforms(cfg['cyclegan']['image_size'])
    tf2 = get_unet_train_transforms(cfg['unet_gan']['image_size'])
    ds  = End2EndDataset(cfg, tf1, tf2)
    loader = DataLoader(ds,
                        batch_size=cfg['unet_gan']['batch_size'],
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    # Models
    unet_params = cfg['unet_gan']['gen_params']
    if isinstance(unet_params.get('features'), list):
        unet_params['features'] = unet_params['features'][0]
    cfg['cyclegan']['gen_params']['norm_layer'] = nn.InstanceNorm2d

    G1 = ResnetGenerator(**cfg['cyclegan']['gen_params']).to(device)
    G2 = ResnetGeneratorUNet(**unet_params).to(device)
    D1 = Discriminator(**cfg['cyclegan']['disc_params']).to(device)
    D2 = Discriminator(**cfg['unet_gan']['disc_params']).to(device)

    # Load best isolated checkpoints (G_best.pth, G_unet_best.pth)
    cyc_path  = os.path.join(cfg['paths']['checkpoints'], 'G_best.pth')
    unet_path = os.path.join(cfg['paths']['unet_gan']['checkpoints'], 'G_unet_best.pth')
    G1.load_state_dict(torch.load(cyc_path,  map_location=device))
    G2.load_state_dict(torch.load(unet_path, map_location=device))

    # Optimizers
    optG  = torch.optim.Adam(list(G1.parameters()) + list(G2.parameters()),
                              lr=cfg['cyclegan']['lr'], betas=(0.5,0.999))
    optD1 = torch.optim.Adam(D1.parameters(),
                              lr=cfg['cyclegan']['lr']*0.5, betas=(0.5,0.999))
    optD2 = torch.optim.Adam(D2.parameters(),
                              lr=cfg['unet_gan']['lr']*0.5, betas=(0.5,0.999))

    # Losses & extractor
    gan_loss = nn.MSELoss().to(device)
    l1_loss  = nn.L1Loss().to(device)
    dino_ext = DINOFeatureExtractor(device=device)

    # Prepare outputs
    visuals_dir = "visuals"
    os.makedirs(visuals_dir, exist_ok=True)
    metrics = []

    # Training
    for epoch in range(cfg['cyclegan']['epochs']):
        t0 = time.time()
        sums = dict(ld1=0, ld2=0, adv1=0, adv2=0, l1=0, s1=0, s2=0, coup=0)
        save_vis = True

        for std, rl, rs in tqdm(loader, desc=f"Epoch {epoch}"):
            # collapse to 1 channel if needed
            if std.dim()==4 and std.size(1)==3:
                std = std.mean(dim=1, keepdim=True)
                rl  = rl.mean(dim=1, keepdim=True)
                rs  = rs.mean(dim=1, keepdim=True)
            std, rl, rs = std.to(device), rl.to(device), rs.to(device)

            fl = G1(std)
            fs = G2(fl)

            # Discriminators
            ld1 = 0.5*(gan_loss(D1(rl), torch.ones_like(D1(rl))) +
                       gan_loss(D1(fl.detach()), torch.zeros_like(D1(fl))))
            ld2 = 0.5*(gan_loss(D2(rs), torch.ones_like(D2(rs))) +
                       gan_loss(D2(fs.detach()), torch.zeros_like(D2(fs))))

            # Generator
            adv1 = gan_loss(D1(fl), torch.ones_like(D1(fl)))
            adv2 = gan_loss(D2(fs), torch.ones_like(D2(fs)))
            l1t  = cfg['cyclegan']['lambda_semantic']*l1_loss(fl, rl) + \
                   cfg['unet_gan']['lambda_l1']      *l1_loss(fs, rs)
            s1   = cfg['cyclegan']['lambda_semantic'] * F.mse_loss(dino_ext(fl), dino_ext(rl))
            s2   = cfg['unet_gan']['lambda_semantic']  * F.mse_loss(dino_ext(fs), dino_ext(rs))
            coup = cfg['coupling']['weight']          * F.mse_loss(dino_ext(fs), dino_ext(std))

            lg = adv1 + adv2 + l1t + s1 + s2 + coup

            # Step discriminators
            optD1.zero_grad(); ld1.backward(); optD1.step()
            optD2.zero_grad(); ld2.backward(); optD2.step()
            # Step generators
            optG.zero_grad(); lg.backward(); optG.step()

            # Accumulate
            for k,v in zip(sums.keys(), [ld1,ld2,adv1,adv2,l1t,s1,s2,coup]):
                sums[k] += v.item()

            # Save one visualization per epoch
            if save_vis:
                inp, low, sub, real = std[:1], fl[:1], fs[:1], rs[:1]
                grid = vutils.make_grid(torch.cat([inp, low, sub, real],0),
                                        nrow=4, normalize=True, scale_each=True)
                vutils.save_image(grid,
                    os.path.join(visuals_dir, f"epoch_{epoch:02d}.png"),
                    nrow=4, normalize=True, scale_each=True)
                save_vis = False

        # Log metrics for this epoch
        N = len(loader)
        elapsed = time.time() - t0
        row = {
            'epoch': epoch,
            'time_s': elapsed,
            **{k: sums[k]/N for k in sums}
        }
        metrics.append(row)
        print(f"[Epoch {epoch:02d}] time={elapsed:.1f}s  D1={row['ld1']:.3f}  D2={row['ld2']:.3f}")

    # Save all metrics to CSV
    csv_path = "end2end_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved metrics to {csv_path}")

    # Save final checkpoints
    out_dir = "data/outputs/checkpoints/end2end"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(G1.state_dict(), os.path.join(out_dir, f"G1_epoch_{epoch}.pth"))
    torch.save(G2.state_dict(), os.path.join(out_dir, f"G2_epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()
