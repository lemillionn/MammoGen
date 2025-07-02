# modules/semantics/feature_extractor.py
import torch
from transformers import AutoImageProcessor, Dinov2Model
from PIL import Image
import torchvision.transforms as T
from core import config

class FeatureExtractor:
    def __init__(self, device='cuda'):
        self.device = device
        cfg = config.load_config()
        # Load processor and model from local cache if specified
        cache_dir = cfg['dinov2'].get('local_cache') or cfg['dinov2']['model_name']
        self.processor = AutoImageProcessor.from_pretrained(
            cache_dir, use_fast=True
        )
        self.model = Dinov2Model.from_pretrained(
            cache_dir
        ).to(self.device)
        self.model.eval()

        # For converting normalized tensors back to PIL
        self.to_pil = T.ToPILImage()

    def __call__(self, image_tensor):
        """
        Accepts a torch tensor image (B, C, H, W) with values in [-1, 1]
        and returns CLS features from DINOv2.
        """
        # Denormalize from [-1,1] to [0,1]
        img = (image_tensor.detach().cpu() + 1.0) * 0.5
        img = img.clamp(0.0, 1.0)

        # Convert each tensor to PIL image
        pil_images = [self.to_pil(img[i]) for i in range(img.shape[0])]

        # Processor will convert to tensors internally
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        )
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token is at position 0
        return outputs.last_hidden_state[:, 0, :]

