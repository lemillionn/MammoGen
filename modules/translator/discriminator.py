import torch
import torch.nn as nn
import torch.nn.utils as utils

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()

        layers = [
            utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False)
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [
            utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False)
            ),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Final output layer: single channel prediction map
        layers += [
            utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
