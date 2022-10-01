from bdb import Breakpoint
import sys, getopt
from datetime import datetime
import time
import cv2
import numpy as np

from torch.utils.data import DataLoader
from dl_ft_1_train import ek_train, collate_fn2
from dl_ft_1_test import ek_test, collate_fn_test

import torch
import torch.nn as nn
import torch.optim as optim
from video_swin_transformer import SwinTransformer3D
from collections import OrderedDict


class VIDEO_SWIN(nn.Module):
    def __init__(self, num_classes = 8, feature_size = 1024):
        super(VIDEO_SWIN, self).__init__()
        #self.backbone  = SwinTransformer3D()

        self.backbone = SwinTransformer3D(embed_dim=128, 
                          depths=[2, 2, 18, 2], 
                          num_heads=[4, 8, 16, 32], 
                          patch_size=(2,4,4), 
                          window_size=(16,7,7), 
                          drop_path_rate=0.4, 
                          patch_norm=True)

        checkpoint = torch.load('./checkpoints/swin_base_patch244_window1677_sthv2.pth')

        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 

        self.backbone.load_state_dict(new_state_dict) 
        self.class_head = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool3d(self.backbone(x),1).flatten(1)
        x = self.class_head(x)
        return x

num_classes = 8
bs = 1
num_epochs = 101
trainKitchen = 'p01'
print("----------------")

#dummy_x = torch.rand(bs, 3, 10, 224, 224).cuda()
model = VIDEO_SWIN(num_classes = num_classes).cuda()
criterion = nn.CrossEntropyLoss()
init_lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=init_lr)
lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
#checkpoint = torch.load('./checkpoints/swin_base_patch244_window1677_sthv2.pth')
#model.load_state_dict(checkpoint, strict=False)

train_dataset = ek_train(shuffle = True, trainKitchen = trainKitchen)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, collate_fn=collate_fn2, drop_last = True)

for epoch in range(num_epochs):
    #if(epoch!=0):
    lr_sched.step()
    for iter, (clip, clip_label,vid_path) in enumerate(train_dataloader):
            clip = torch.swapaxes(clip, 1, 2)
            clip = clip.cuda()

            optimizer.zero_grad()

            preds = model(clip)

            clip_label = torch.as_tensor(clip_label).cuda()
            loss = criterion(preds, clip_label)
            
            #backward pass
            loss.backward()

            #update parameters
            optimizer.step()