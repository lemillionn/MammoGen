import os
import random
import warnings
from PIL import Image
from torch.utils.data import Dataset

def get_view_from_filename(filename):
    """
    Extracts the view token from filenames like 'P1_L_DM_MLO.png' → 'MLO'
    """
    base = os.path.basename(filename)
    return base.split('_')[-1].split('.')[0]

class UnpairedImageDataset(Dataset):
    """
    Unpaired dataset for CycleGAN:
    - Tries to pair by view if possible.
    - If no views match, falls back to random unpaired sampling.
    """
    def __init__(self, root_low, root_rsna, transform=None):
        self.transform   = transform
        self.low_images  = []
        self.rsna_images = []
        self.low_by_view = {}
        self.rsna_by_view= {}

        # collect low-energy images
        for fn in os.listdir(root_low):
            if fn.lower().endswith(('.png','jpg','jpeg')):
                p = os.path.join(root_low, fn)
                self.low_images.append(p)
                v = get_view_from_filename(fn)
                self.low_by_view.setdefault(v, []).append(p)

        # collect RSNA images
        for fn in os.listdir(root_rsna):
            if fn.lower().endswith(('.png','jpg','jpeg')):
                p = os.path.join(root_rsna, fn)
                self.rsna_images.append(p)
                v = get_view_from_filename(fn)
                self.rsna_by_view.setdefault(v, []).append(p)

        # identify common views
        self.views = list(set(self.low_by_view) & set(self.rsna_by_view))
        if self.views:
            # length = sum of min-counts per view
            self.length = sum(
                min(len(self.low_by_view[v]), len(self.rsna_by_view[v]))
                for v in self.views
            )
        else:
            warnings.warn(
                f"No common views between:\n • {root_low}\n • {root_rsna}\n"
                "→ falling back to fully random pairing."
            )
            if not self.low_images or not self.rsna_images:
                raise RuntimeError("No images found for fallback pairing.")
            # arbitrary epoch length = larger folder size
            self.length = max(len(self.low_images), len(self.rsna_images))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.views:
            v = random.choice(self.views)
            low_p  = random.choice(self.low_by_view[v])
            rsna_p = random.choice(self.rsna_by_view[v])
        else:
            low_p  = random.choice(self.low_images)
            rsna_p = random.choice(self.rsna_images)

        img_low  = Image.open(low_p).convert("RGB")
        img_rsna = Image.open(rsna_p).convert("RGB")

        if self.transform:
            img_low  = self.transform(img_low)
            img_rsna = self.transform(img_rsna)

        return img_low, img_rsna
