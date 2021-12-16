from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import pdb

class UNet3dLite(nn.Module):
    def __init__(self):
        super().__init__()

        # self.down = nn.MaxPool3d((1, 2, 2))

        self.conv0 = nn.Conv3d(1, 32, (3, 3, 3), padding=1)
        self.conv1 = nn.Conv3d(32, 32, (3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(32, 64, (3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(64, 64, (3, 3, 3), padding=1)
        self.conv4 = nn.Conv3d(64, 128, (3, 3, 3), padding=1)
        self.conv5 = nn.Conv3d(128, 128, (3, 3, 3), padding=1)
        self.conv6 = nn.Conv3d(128, 256, (3, 3, 3), padding=1)
        self.conv7 = nn.Conv3d(256, 128, (3, 3, 3), padding=1)

        self.upconv0 = nn.ConvTranspose3d(128, 512, (2, 2, 2), (2, 2, 2))
        self.mconv0 = nn.Conv3d(640, 256, (3, 3, 3), padding=1)
        self.mconv1 = nn.Conv3d(256, 64, (3, 3, 3), padding=1)

        self.upconv1 = nn.ConvTranspose3d(64, 256, (2, 2, 2), (2, 2, 2))
        self.mconv2 = nn.Conv3d(320, 128, (3, 3, 3), padding=1)
        self.mconv3 = nn.Conv3d(128, 32, (3, 3, 3), padding=1)

        self.upconv2 = nn.ConvTranspose3d(32, 128, (2, 2, 2), (2, 2, 2))
        self.mconv4 = nn.Conv3d(160, 64, (3, 3, 3), padding=1)
        self.mconv5 = nn.Conv3d(64, 64, (3, 3, 3), padding=1)

        self.conv_final = nn.Conv3d(64, 1, 1)

        self.down = nn.MaxPool3d((2,2,2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        conv0 = F.relu(self.conv0(inp))
        conv1 = F.relu(self.conv1(conv0))
        down0 = self.down(conv1)
        conv2 = F.relu(self.conv2(down0))
        conv3 = F.relu(self.conv3(conv2))
        down1 = self.down(conv3)
        conv4 = F.relu(self.conv4(down1))
        conv5 = F.relu(self.conv5(conv4))
        down2 = self.down(conv5)
        conv6 = F.relu(self.conv6(down2))
        conv7 = F.relu(self.conv7(conv6))

        upconv0 = F.relu(self.upconv0(conv7))
        d0, u0 = conv5, upconv0
        mrg0 = torch.cat((d0, u0), 1)
        mconv0 = F.relu(self.mconv0(mrg0))
        mconv1 = F.relu(self.mconv1(mconv0))

        upconv1 = F.relu(self.upconv1(mconv1))
        d1, u1 = conv3, upconv1
        mrg1 = torch.cat((d1, u1), 1)
        mconv2 = F.relu(self.mconv2(mrg1))
        mconv3 = F.relu(self.mconv3(mconv2))

        upconv2 = F.relu(self.upconv2(mconv3))
        d2, u2 = conv1, upconv2
        mrg2 = torch.cat((d2, u2), 1)
        mconv4 = F.relu((self.mconv4(mrg2)))
        mconv5 = F.relu(self.mconv5(mconv4))

        out = self.sigmoid(self.conv_final(mconv5))

        return out


if __name__ == '__main__':
    m = UNet3dLite()
    x = torch.randn(6, 1, 64, 128, 128)
    y = m(x)
    print(y.shape)