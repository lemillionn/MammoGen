import torch
import torch.nn as nn

class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, features_real, features_fake):
        """
        Computes the MSE loss between the feature maps of real and fake images.

        Args:
            features_real (Tensor): Feature map from real image
            features_fake (Tensor): Feature map from generated image

        Returns:
            Tensor: Scalar loss
        """
        return self.criterion(features_real, features_fake)
