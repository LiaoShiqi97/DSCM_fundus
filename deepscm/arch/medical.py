from torch import nn

import numpy as np

image_channels= 3
# note cur_channels= 1 for gray image


class Encoder(nn.Module):
    def __init__(self, num_convolutions=1, filters=(16, 24, 32, 64, 128), latent_dim: int = 128, input_size=(image_channels, 128, 128)):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim

        layers = []

        cur_channels = image_channels

        for c in filters:
            for _ in range(0, num_convolutions - 1):
                layers += [nn.Conv2d(cur_channels, c, 3, 1, 1), nn.BatchNorm2d(c), nn.LeakyReLU(.1, inplace=True)]
                cur_channels = c

            layers += [nn.Conv2d(cur_channels, c, 4, 2, 1), nn.BatchNorm2d(c), nn.LeakyReLU(.1, inplace=True)]

            cur_channels = c

        self.cnn = nn.Sequential(*layers)

        self.intermediate_shape = np.array(input_size) // (2 ** len(filters))

        self.intermediate_shape[0] = cur_channels

        self.fc = nn.Sequential(
            nn.Linear(np.prod(self.intermediate_shape), latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(.1, inplace=True)
        )

    def forward(self, x):

        x = self.cnn(x).view(-1, np.prod(self.intermediate_shape))
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, num_convolutions=1, filters=(128, 64, 32, 24, 16), latent_dim: int = 128, output_size=(image_channels, 128, 128), upconv=False):
        super().__init__()

        self.num_convolutions = num_convolutions
        self.filters = filters
        self.latent_dim = latent_dim

        self.intermediate_shape = np.array(output_size) // (2 ** (len(filters)-1))
        self.intermediate_shape[0] = filters[0]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, np.prod(self.intermediate_shape)),
            nn.BatchNorm1d(np.prod(self.intermediate_shape)),
            nn.LeakyReLU(.1, inplace=True)
        )

        layers = []

        cur_channels = filters[0]
        for c in filters[1:]:
            for _ in range(0, num_convolutions - 1):
                layers += [nn.Conv2d(cur_channels, cur_channels, 3, 1, 1), nn.BatchNorm2d(cur_channels), nn.LeakyReLU(.1, inplace=True)]

            if upconv:
                layers += [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(cur_channels, c, kernel_size=5, stride=1, padding=2)
                ]
            else:
                layers += [nn.ConvTranspose2d(cur_channels, c, kernel_size=4, stride=2, padding=1)]
            layers += [nn.BatchNorm2d(c), nn.LeakyReLU(.1, inplace=True)]

            cur_channels = c

        # layers += [nn.Conv2d(cur_channels, image_channels, 1, 1)]
        # change the the output from [batch, 16, 128, 128] to [batch, image_channels, 128, 128]
        # so the conditional affinetransformation is decoder = Conv2dIndepNormal(decoder, image_channels, image_channels)
        # or comment this, then conditional affinetransformation is decoder = Conv2dIndepNormal(decoder, 16, image_channels)
        # we comment this as we think this is channel reduction makes no sense and can lose information.


        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x).view(-1, *self.intermediate_shape)


        return self.cnn(x)


