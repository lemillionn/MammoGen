# modules/semantics/dino_feature.py
import torch
import timm
import torch.nn as nn

def load_dino_model(device="cuda"):
    model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
    model.eval().to(device)
    return model

def extract_cls_token(features):
    """Extract the [CLS] token from ViT output."""
    return features[:, 0]

class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, real_features, generated_features):
        real_cls = extract_cls_token(real_features)
        gen_cls = extract_cls_token(generated_features)
        return self.criterion(real_cls, gen_cls)
