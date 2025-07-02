from PIL import Image
import torch
from torchvision import transforms
import os

def load_image_tensor(path, image_size=256, device='cpu', force_grayscale=False):
    """
    Load image from path as normalized tensor on device.
    Supports grayscale or RGB based on force_grayscale flag.
    Returns: Tensor of shape (1, C, H, W) in range [-1, 1]
    """
    image = Image.open(path)

    if force_grayscale:
        image = image.convert('L')  # grayscale
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else:
        image = image.convert('RGB')  # 3 channels
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    tensor = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)
    return tensor.float()


def save_image_tensor(tensor, path):
    """
    Save tensor image to disk (unnormalizing first).
    Automatically handles grayscale (1 channel) and RGB (3 channel).
    """
    from torchvision.utils import save_image

    tensor = tensor.squeeze(0).cpu()  # (C, H, W)

    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with shape (C, H, W), got {tensor.shape}")

    c = tensor.shape[0]
    if c == 1:
        unnormalize = transforms.Normalize(mean=[-1.0], std=[2.0])
    elif c == 3:
        unnormalize = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
    else:
        raise ValueError(f"Unsupported number of channels: {c}")

    img_tensor = unnormalize(tensor)
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(img_tensor, path)


def load_mask(mask_path, image_size=256, device='cpu'):
    """
    Load segmentation mask as a binary tensor on device.
    Returns: Tensor of shape (1, 1, H, W) with values in {0.0, 1.0}
    """
    mask = Image.open(mask_path).convert('L')  # grayscale
    mask = mask.resize((image_size, image_size))
    tensor = transforms.ToTensor()(mask)  # (1, H, W)
    binary = (tensor > 0.5).float()
    return binary.unsqueeze(0).to(device)  # (1, 1, H, W)
