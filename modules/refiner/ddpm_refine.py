import os
from typing import Union

import torch
from torch import Tensor
from torchvision import transforms
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler


class DDPMRefiner:
    def __init__(self, model: UNet2DModel, scheduler: DDPMScheduler, device: torch.device):
        self.device = device
        self.model = model.to(device)       # Force model to correct device
        self.scheduler = scheduler          # Scheduler is usually device-independent

    def refine(self, img: Union[Tensor, Image.Image]) -> Tensor:
        if isinstance(img, Image.Image):
            tensor = transforms.ToTensor()(img).unsqueeze(0)  # [1, C, H, W]
        elif isinstance(img, torch.Tensor):
            if img.dim() == 3:
                tensor = img.unsqueeze(0)
            elif img.dim() == 4:
                tensor = img
            else:
                raise ValueError(f"Unexpected tensor shape: {img.shape}")
        else:
            raise TypeError(f"Unsupported input type: {type(img)}")

        tensor = tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            noise = torch.randn_like(tensor).to(self.device)
            timesteps = torch.tensor([0], dtype=torch.long, device=self.device)
            noisy = self.scheduler.add_noise(tensor, noise, timesteps)
            output = self.model(noisy, timesteps)
            pred = output.sample

        return pred.squeeze(0).cpu().clamp(-1, 1)


if __name__ == "__main__":
    import argparse
    from torchvision.transforms import ToPILImage
    from core.image_io import load_image_tensor
    from diffusers.utils import logging

    parser = argparse.ArgumentParser(description="Refine an image using a DDPM model.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save the refined image.")
    parser.add_argument("--model-id", type=str, default="data/outputs/ddpm_model", help="Path to your trained DDPM model.")
    args = parser.parse_args()

    # Setup
    logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load input image
    img = load_image_tensor(args.input, device=device)

    # Load your trained model
    model = UNet2DModel.from_pretrained(args.model_id)
    scheduler = DDPMScheduler.from_pretrained(args.model_id)

    # Refine
    refiner = DDPMRefiner(model.to(device), scheduler, device)
    refined = refiner.refine(img)

    # Save as grayscale
    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "refined_sample.png")

    refined_img = refined.squeeze()
    if refined_img.ndim == 3:
        refined_img = refined_img.squeeze(0)
    ToPILImage()(refined_img).save(save_path)

    print(f"✅ Saved refined image to {save_path}")
