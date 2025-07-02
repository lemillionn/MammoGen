import torch
from modules.semantics.feature_extractor import FeatureExtractor
from modules.semantics.feature_loss import FeatureLoss

class SemanticLossWrapper:
    def __init__(self, device):
        self.extractor = FeatureExtractor(device)
        self.loss_fn = FeatureLoss()

    def compute_loss(self, real_image, fake_image):
        with torch.no_grad():
            feat_real = self.extractor(real_image)
            feat_fake = self.extractor(fake_image)
        return self.loss_fn(feat_real, feat_fake)
