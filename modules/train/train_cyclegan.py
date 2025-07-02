import glob
import os
import re
import csv
import itertools
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from core import config
from core.transforms import get_train_transforms
from modules.dataloader.dataloader_cyclegan import UnpairedImageDataset
from modules.translator.generator import Generator
from modules.translator.discriminator import Discriminator
from modules.translator.gan_infer import save_sample_images
from modules.semantics.semantic_loss_wrapper import SemanticLossWrapper

def compute_gradient_penalty(D, real, fake, device):
    α = torch.rand(real.size(0), 1, 1, 1, device=device)
    inter = (α * real + (1 - α) * fake).requires_grad_(True)
    out = D(inter)
    grad = torch.autograd.grad(
        outputs=out, inputs=inter,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True
    )[0]
    return ((grad.view(grad.size(0), -1).norm(2, 1) - 1) ** 2).mean()

def get_latest_checkpoint_index(ckpt_dir, prefix):
    files = glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pth"))
    indices = [
        int(re.search(r"epoch_(\d+)", f).group(1))
        for f in files if re.search(r"epoch_(\d+)", f)
    ]
    return max(indices) if indices else -1

def train_cyclegan():
    cfg    = config.load_config()
    device = torch.device(cfg['cyclegan']['device'])

    # — Data loader —
    tf = get_train_transforms(cfg['cyclegan']['image_size'])
    ds = UnpairedImageDataset(
        cfg['paths']['unpaired_low'],
        cfg['paths']['unpaired_rsna'],
        transform=tf
    )
    if cfg['cyclegan'].get('debug_max_samples'):
        cnt = min(len(ds), cfg['cyclegan']['debug_max_samples'])
        ds = Subset(ds, list(range(cnt)))
    dl = DataLoader(
        ds,
        batch_size=cfg['cyclegan']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # — Models & semantic wrapper —
    G, F = Generator().to(device), Generator().to(device)
    Dx, Dy = Discriminator().to(device), Discriminator().to(device)
    sem = SemanticLossWrapper(device)

    # — Opts & schedulers —
    g_opt = torch.optim.Adam(
        itertools.chain(G.parameters(), F.parameters()),
        lr=cfg['cyclegan']['lr'], betas=(0.5, 0.999)
    )
    dx_opt = torch.optim.Adam(Dx.parameters(),
                              lr=cfg['cyclegan']['lr'], betas=(0.5, 0.999))
    dy_opt = torch.optim.Adam(Dy.parameters(),
                              lr=cfg['cyclegan']['lr'], betas=(0.5, 0.999))

    def lr_lambda(e):
        half = cfg['cyclegan']['epochs'] // 2
        return 1.0 if e < half else max(0, 1 - (e - half) / (cfg['cyclegan']['epochs'] - half))

    g_sch  = torch.optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=lr_lambda)
    dx_sch = torch.optim.lr_scheduler.LambdaLR(dx_opt, lr_lambda=lr_lambda)
    dy_sch = torch.optim.lr_scheduler.LambdaLR(dy_opt, lr_lambda=lr_lambda)

    # — Losses —
    adv = torch.nn.MSELoss()
    cyc = torch.nn.L1Loss()
    λgp = cfg['cyclegan']['lambda_gp']
    λid = cfg['cyclegan']['lambda_identity']

    # — Prepare outputs & metrics —
    os.makedirs(cfg['paths']['checkpoints'], exist_ok=True)
    os.makedirs(cfg['paths']['output_samples'], exist_ok=True)
    metrics = []
    best_G_loss = float('inf')

    # — Resume checkpoint if exists —
    start_epoch = get_latest_checkpoint_index(cfg['paths']['checkpoints'], "G") + 1
    if start_epoch > 0:
        print(f"🔁 Resuming from epoch {start_epoch}")
        G.load_state_dict(torch.load(
            os.path.join(cfg['paths']['checkpoints'], f"G_epoch_{start_epoch-1}.pth"),
            map_location=device))
        F.load_state_dict(torch.load(
            os.path.join(cfg['paths']['checkpoints'], f"F_epoch_{start_epoch-1}.pth"),
            map_location=device))

    # — Training loop —
    for epoch in range(start_epoch, cfg['cyclegan']['epochs']):
        print(f"Epoch {epoch+1}/{cfg['cyclegan']['epochs']}")
        sum_lossG = sum_lossDx = sum_lossDy = 0.0
        n_batches = 0

        for i, (low, rsna) in enumerate(tqdm(dl)):
            low, rsna = low.to(device), rsna.to(device)
            n_batches += 1

            # — Generator step —
            g_opt.zero_grad()
            fr = G(low)
            rl = F(fr)
            fl = F(rsna)
            rr = G(fl)
            id_loss = λid * (cyc(G(rsna), rsna) + cyc(F(low), low))
            lossG = (
                adv(Dx(fr), torch.ones_like(Dx(fr)) * 0.9) +
                adv(Dy(fl), torch.ones_like(Dy(fl)) * 0.9) +
                cfg['cyclegan']['lambda_cycle'] * (cyc(rl, low) + cyc(rr, rsna)) +
                cfg['cyclegan']['lambda_semantic'] * sem.compute_loss(rsna, fr) +
                id_loss
            )
            lossG.backward()
            g_opt.step()

            # — Detach for D steps —
            fr_det = fr.detach()
            fl_det = fl.detach()

            # — Discriminator X step —
            dx_opt.zero_grad()
            real_x = Dx(rsna)
            fake_x = Dx(fr_det)
            lossDx = (
                0.5 * (adv(real_x, torch.ones_like(real_x)) +
                       adv(fake_x, torch.zeros_like(fake_x))) +
                λgp * compute_gradient_penalty(Dx, rsna, fr_det, device)
            )
            lossDx.backward()
            dx_opt.step()

            # — Discriminator Y step —
            dy_opt.zero_grad()
            real_y = Dy(low)
            fake_y = Dy(fl_det)
            lossDy = (
                0.5 * (adv(real_y, torch.ones_like(real_y)) +
                       adv(fake_y, torch.zeros_like(fake_y))) +
                λgp * compute_gradient_penalty(Dy, low, fl_det, device)
            )
            lossDy.backward()
            dy_opt.step()

            # — Save sample images —
            if i % cfg['cyclegan']['sample_interval'] == 0:
                save_sample_images(
                    low, fr, rsna, fl,
                    epoch, i,
                    cfg['paths']['output_samples']
                )

            sum_lossG  += lossG.item()
            sum_lossDx += lossDx.item()
            sum_lossDy += lossDy.item()

        # — End epoch —
        g_sch.step(); dx_sch.step(); dy_sch.step()

        avgG  = sum_lossG  / n_batches
        avgDx = sum_lossDx / n_batches
        avgDy = sum_lossDy / n_batches
        metrics.append({
            'epoch': epoch,
            'loss_G':  avgG,
            'loss_Dx': avgDx,
            'loss_Dy': avgDy
        })
        print(f"  Avg G: {avgG:.4f}, Dx: {avgDx:.4f}, Dy: {avgDy:.4f}")

        # — Save best generators —
        if avgG < best_G_loss:
            best_G_loss = avgG
            torch.save(G.state_dict(),
                       os.path.join(cfg['paths']['checkpoints'], 'G_best.pth'))
            torch.save(F.state_dict(),
                       os.path.join(cfg['paths']['checkpoints'], 'F_best.pth'))

        # — Save epoch checkpoints —
        torch.save(G.state_dict(),
                   os.path.join(cfg['paths']['checkpoints'], f"G_epoch_{epoch}.pth"))
        torch.save(F.state_dict(),
                   os.path.join(cfg['paths']['checkpoints'], f"F_epoch_{epoch}.pth"))

    # — Save metrics to CSV —
    with open('cyclegan_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    print("✅ CycleGAN training complete.")

if __name__ == "__main__":
    train_cyclegan()
