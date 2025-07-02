import os, glob, numpy as np, torch
from PIL import Image
from torch.utils.data import Dataset

class PairedImageDataset(Dataset):
    """
    Paired low-energy / subtracted-image dataset for UNet.
    Loads RGB and applies the same Albumentations to both.
    """
    def __init__(self, dir_low, dir_high, transform=None):
        self.transform = transform
        self.low_paths  = sorted(glob.glob(os.path.join(dir_low, '*')))
        self.high_paths = []
        for p in self.low_paths:
            fname = os.path.basename(p)
            target_name = fname.replace('_DM_','_CM_') if '_DM_' in fname else fname
            tp = os.path.join(dir_high, target_name)
            if not os.path.exists(tp):
                raise FileNotFoundError(f"No paired file for {fname} in {dir_high}")
            self.high_paths.append(tp)

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        # load as RGB
        low  = np.array(Image.open(self.low_paths[idx]).convert('RGB'))
        high = np.array(Image.open(self.high_paths[idx]).convert('RGB'))

        if self.transform:
            aug = self.transform(image=low, target=high)
            return aug['image'], aug['target']

        # fallback: [H,W,3] → [3,H,W] in [0,1]
        low_t  = torch.from_numpy(low ).permute(2,0,1).float() / 255.0
        high_t = torch.from_numpy(high).permute(2,0,1).float() / 255.0
        return low_t, high_t
