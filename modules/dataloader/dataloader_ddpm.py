# modules/dataloader/dataloader_ddpm.py

import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    """
    A simple paired dataset that zips together
    all images in image_dir with all images in label_dir
    via sorted order.
    """

    def __init__(self, image_dir: str, label_dir: str, transform=None):
        self.transform = transform

        # collect and sort all image paths
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.[jJpP][pPnN][gG]"))
        )
        self.label_paths = sorted(
            glob.glob(os.path.join(label_dir, "*.[jJpP][pPnN][gG]"))
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in image_dir = {image_dir}")
        if len(self.label_paths) == 0:
            raise RuntimeError(f"No images found in label_dir = {label_dir}")
        if len(self.image_paths) != len(self.label_paths):
            raise RuntimeError(
                f"Number of images ({len(self.image_paths)}) "
                f"does not match number of labels ({len(self.label_paths)})"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load paired images and convert to grayscale
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        img = Image.open(img_path).convert("L")  # Grayscale
        lbl = Image.open(lbl_path).convert("L")  # Grayscale

        if self.transform is not None:
            img = self.transform(img)
            lbl = self.transform(lbl)

        return {"input": img, "target": lbl}
