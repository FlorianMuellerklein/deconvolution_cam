import torch
import torchvision
import torch.nn as nn
import torchvision as vsn
import torch.nn.functional as F
import torchvision.transforms as transforms


class VGGlike_vanilla(nn.Module):
    def __init__(self):
        super(VGGlike_vanilla, self).__init__()
        # set up the conv layers
        self.features = nn.Sequential(
            nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.MaxPool2d(kernel_size=2),
            # 32x64x64
            nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.MaxPool2d(kernel_size=2),
            # 64x32x32
            nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.MaxPool2d(kernel_size=2),
            # 128x16x16
            nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.MaxPool2d(kernel_size=2),
            # 256x8x8
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1),
            nn.ELU(True),
            # 512x8x8
        )

        self.classifier = nn.Sequential(
            #nn.Conv2d(256, 20, kernel_size=3, stride=1, padding=1),
            nn.Linear(512, 20)
        )

        self.avgpool = nn.AvgPool2d(8)


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        
        return out

class VGGlike_upsample(nn.Module):
    def __init__(self):
        super(VGGlike_upsample, self).__init__()
        self.conv1 = nn.Conv2d(3,32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)

        self.conv8 = nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)

        self.lastdeconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ELU(True)
        )

        self.elu = nn.ELU(True)

        self.dense = nn.Linear(128,20)
        self.avgpool = nn.AvgPool2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        c1 = self.elu(self.conv1(x))
        c2 = self.elu(self.conv2(c1))
        mp1 = self.maxpool(c2)
        # 32x64x64
        c3 = self.elu(self.conv3(mp1))
        c4 = self.elu(self.conv4(c3))
        mp2 = self.maxpool(c4)
        # 64x32x32
        c5 = self.elu(self.conv5(mp2))
        c6 = self.elu(self.conv6(c5))
        c7 = self.elu(self.conv7(c6))
        mp3 = self.maxpool(c7)
        # 128x16x16
        c8 = self.elu(self.conv8(mp3))
        c9 = self.elu(self.conv9(c8))
        c10 = self.elu(self.conv10(c9))
        mp4 = self.maxpool(c10)
        # 256x8x8
        c11 = self.elu(self.conv11(mp4))
        c12 = self.elu(self.conv12(c11))
        c13 = self.elu(self.conv13(c12))
        # 512x8x8
        d1 = self.elu(self.deconv1(c13))
        #print(c10.size(), d1.size())
        d1c = c10 + d1
        # 256x16x16
        d2 = self.lastdeconv(d1c)
        # 128x32x32

        gap = self.avgpool(d2)
        out = self.dense(gap.view(gap.size(0), -1))

        return out
