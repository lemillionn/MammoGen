import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_unet_train_transforms(image_size: int):
    """
    Paired transforms for UNet GAN on RGB mammograms.
    - Resize both low & high to (image_size, image_size)
    - Normalize to [-1,1]
    - Convert to a 3×H×W torch.Tensor
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            ToTensorV2(),
        ],
        additional_targets={'target': 'image'},
        is_check_shapes=False,
    )

def get_unet_test_transforms(image_size: int):
    """Same as train (resize & normalize)."""
    return get_unet_train_transforms(image_size)
