import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from core.config import load_config
from core.transforms import get_eval_transforms
from modules.dataloader.dataloader_paired import PairedImageDataset
from modules.translator.generator_unet import UNetGenerator


def save_unet_gan_samples(input_images, output_images, target_images,
                          epoch, batch, save_dir,
                          prefix_types=('real_low','fake_subtracted','real_subtracted')):
    """
    Save a batch of training samples: real low-energy, fake subtracted, real subtracted.
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx, (inp, out, tgt) in enumerate(zip(input_images, output_images, target_images)):
        base = f"epoch{epoch}_batch{batch}_img{idx}"
        save_image(inp, os.path.join(save_dir, f"{base}_{prefix_types[0]}.png"))
        save_image(out, os.path.join(save_dir, f"{base}_{prefix_types[1]}.png"))
        save_image(tgt, os.path.join(save_dir, f"{base}_{prefix_types[2]}.png"))


def save_unet_gan_batch(model, dataloader, device, save_dir,
                        prefix_types=('real_low','fake_subtracted','real_subtracted')):
    """
    Run full-dataset inference and save all samples in batches.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (low, sub) in enumerate(dataloader):
            low, sub = low.to(device), sub.to(device)
            fake_sub = model(low)
            for idx, (inp, out, tgt) in enumerate(zip(low, fake_sub, sub)):
                base = f"batch{batch_idx}_img{idx}"
                save_image(inp, os.path.join(save_dir, f"{base}_{prefix_types[0]}.png"))
                save_image(out, os.path.join(save_dir, f"{base}_{prefix_types[1]}.png"))
                save_image(tgt, os.path.join(save_dir, f"{base}_{prefix_types[2]}.png"))


def infer_unet_gan(ckpt_path, data_root, output_dir, batch_size=1):
    cfg = load_config()
    device = torch.device(cfg['unet_gan']['device'])

    # Load generator
    G = UNetGenerator(in_channels=3, out_channels=3).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device))

    # Dataset & loader
    tf = get_eval_transforms(cfg['unet_gan']['image_size'])
    ds = PairedImageDataset(
        os.path.join(data_root, 'low'),
        os.path.join(data_root, 'subtracted'),
        transform=tf
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Run inference & save
    save_unet_gan_batch(G, dl, device, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     required=True, help='path to G checkpoint')
    parser.add_argument('--data_root',required=True, help='root dataset folder')
    parser.add_argument('--out',      required=True, help='output samples folder')
    parser.add_argument('--bs',       type=int, default=1)
    args = parser.parse_args()

    infer_unet_gan(args.ckpt, args.data_root, args.out, args.bs)
