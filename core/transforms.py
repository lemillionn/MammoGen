from torchvision import transforms

def get_train_transforms(image_size):
    """
    Transforms for training data:
    - Resize to configured size
    - Random horizontal flip (augment unpaired data)
    - Convert to tensor
    - Normalize to [-1, 1] range
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_eval_transforms(image_size):
    """
    Transforms for evaluation data:
    - Resize to configured size
    - Convert to tensor
    - Normalize same as training
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

