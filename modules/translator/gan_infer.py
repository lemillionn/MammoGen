import os
import torch
from torchvision import transforms
from PIL import Image
from modules.translator.generator import Generator
import torchvision.utils as vutils


def run_gan_inference(input_path, model_path, output_path, device='cuda'):
    # Load generator model
    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load and transform image
    img = Image.open(input_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Generate image
    with torch.no_grad():
        fake_tensor = model(img_tensor)

    # Save generated image
    fake_image = transforms.ToPILImage()(fake_tensor.squeeze().cpu())
    fake_image.save(output_path)


from torchvision.transforms import ToPILImage

def save_sample_images(real_low, fake_rsna, real_rsna, fake_low, epoch, batch_idx, out_dir):
    """
    Save individual PNGs of one example per batch for visualization during training.

    Args:
        real_low:   Tensor(B,C,H,W) real low-energy batch
        fake_rsna:  Tensor(B,C,H,W) generated RSNA batch
        real_rsna:  Tensor(B,C,H,W) real RSNA batch
        fake_low:   Tensor(B,C,H,W) generated low-energy batch
        epoch:      int current epoch
        batch_idx:  int current batch index
        out_dir:    str output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    to_pil = ToPILImage()

    # pick the first sample in the batch
    rl = to_pil(real_low[0].cpu().clamp(-1,1).add(1).div(2))
    fr = to_pil(fake_rsna[0].cpu().clamp(-1,1).add(1).div(2))
    rr = to_pil(real_rsna[0].cpu().clamp(-1,1).add(1).div(2))
    fl = to_pil(fake_low[0].cpu().clamp(-1,1).add(1).div(2))

    rl.save(os.path.join(out_dir, f"epoch{epoch+1}_batch{batch_idx}_real_low.png"))
    fr.save(os.path.join(out_dir, f"epoch{epoch+1}_batch{batch_idx}_fake_rsna.png"))
    rr.save(os.path.join(out_dir, f"epoch{epoch+1}_batch{batch_idx}_real_rsna.png"))
    fl.save(os.path.join(out_dir, f"epoch{epoch+1}_batch{batch_idx}_fake_low.png"))
