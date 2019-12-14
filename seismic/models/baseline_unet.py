import torch
import torch.nn as nn
from torchvision.models import resnet34

from seismic.models.layers import (DecoderSEBlockV2, ConvRelu)

class UnetResnet34(nn.Module):
    def __init__(self, num_classes=1, num_filters=16, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        encoder = resnet34(pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.ModuleList([
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
                self.pool),
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4])

        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(512, 1)

        self.center = DecoderSEBlockV2(512, num_filters * 8 * 2, num_filters * 8)

        self.decoder = nn.ModuleList([
            DecoderSEBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8),
            DecoderSEBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2),
            DecoderSEBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2),
        ])

        self.dec1 = DecoderSEBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        encoder_results = []
        for stage in self.encoder:
            x = stage(x)
            encoder_results.append(x.clone())

        x = self.center(self.pool(x))
        # x = self.center(x)

        for i, decoder in enumerate(self.decoder):
            x = self.decoder[i](torch.cat([x, encoder_results[-i - 1]], 1))

        x = self.dec1(x)
        x = self.dec0(x)
        x = self.final(x)

        return x
