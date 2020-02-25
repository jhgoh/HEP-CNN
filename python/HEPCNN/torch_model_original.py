#!/usr/bin/env python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, width, height, **kwargs):
        super(MyModel, self).__init__()
        self.nch = 3
        self.fw = width
        self.fh = height

        self.conv = []

        self.conv.extend([
            nn.Conv2d(self.nch, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
        ])
        self.fw, self.fh = self.fw//2, self.fh//2

        self.conv.extend([
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
        ])
        self.fw, self.fh = (self.fw)//2, (self.fh)//2
        self.fw, self.fh = (self.fw)//2, (self.fh)//2

        self.conv.extend([
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
        ])
        self.fw, self.fh = self.fw//2, self.fh//2

        if width > 128:
            self.conv.extend([
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=1),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.ReLU(),
            ])
            self.fw, self.fh = (self.fw)//2, (self.fh)//2

        self.conv = nn.Sequential(*self.conv)

        self.fc = nn.Sequential(
            nn.Linear(self.fw*self.fh*256, 512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.fw*self.fh*256)
        x = self.fc(x)

        return x


