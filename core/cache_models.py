# cache_models.py

from transformers import AutoImageProcessor, Dinov2Model
from diffusers import DDPMPipeline
import os

# create cache directories
os.makedirs("core/cache/dinov2-base", exist_ok=True)
os.makedirs("core/cache/ddpm-cifar10-32", exist_ok=True)

# 1) Cache DINOv2
dinov2_dir = "core/cache/dinov2-base"
print(f"Saving DINOv2 to {dinov2_dir}…")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model     = Dinov2Model.from_pretrained("facebook/dinov2-base")
processor.save_pretrained(dinov2_dir)
model.save_pretrained(dinov2_dir)

# 2) Cache DDPM pipeline
ddpm_dir = "core/cache/ddpm-cifar10-32"
print(f"Saving DDPM pipeline to {ddpm_dir}…")
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
pipe.save_pretrained(ddpm_dir)

print("✅ Done caching both models.")
