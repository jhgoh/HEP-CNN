#!/usr/bin/env python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, width, height, model='default'):
        super(MyModel, self).__init__()
        self.fw = width//2//2//2
        self.fh = height//2//2//2

        self.nch = 5 if '5ch' in model else 3
        self.doLog = ('log' in model)

        self.conv = nn.Sequential(
            nn.Conv2d(self.nch, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(64, 128, kernel_size=(2, 2), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            nn.Conv2d(128, 256, kernel_size=(2, 2), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
            nn.Dropout2d(0.5),

            #nn.Conv2d(256, 256, kernel_size=(3, 3), stride=2, padding=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256, eps=0.001, momentum=0.99),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fw*self.fh*256, 512),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.nch == 5:
            xx = x[:,:2,:,:]
            x = torch.cat((x, xx), dim=1)
        if self.doLog:
            x[:,:2,:,:] = torch.log10(x[:,:2,:,:]/1e-5+1)
        x = self.conv(x)
        x = x.view(-1, self.fw*self.fh*256)
        x = self.fc(x)
        return x


