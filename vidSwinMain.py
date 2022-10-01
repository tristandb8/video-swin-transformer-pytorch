#from ast import Num
#import torch
#import torch.nn as nn
#from video_swin_transformer import SwinTransformer3D

#model = SwinTransformer3D()
#num_classes = 8
# model = nn.Sequential(model, nn.Linear(768, num_classes))
#print(model)

#dummy_x = torch.rand(1, 3, 10, 224, 224)
#logits = model(dummy_x)
#print(logits.shape) # torch.Size([1, 768, 3, 7, 7]) [batch, c, time, h, w]
#logits1 = torch.nn.functional.adaptive_avg_pool3d(logits, 1)
#print(logits1.shape)
#logits1 = logits1.flatten(1)
#print(logits1.shape) #

#from numpy import vstack
#from numpy import argmax
#from sklearn.metrics import accuracy_score
import numpy as np
import sys, getopt


import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append('/home/tr248228/RP_EvT/October/n-epic-kitchens')
from dl_ft_1_train import ek_train, collate_fn2
from dl_ft_1_test import ek_test, collate_fn_test
from functions_NEK import train_accuracy, validate, train_epoch, getInputs, startLog


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




def main(argv):
    trainKitchen, testKitchen, logFolder, note = getInputs(argv)
    logFile = startLog(trainKitchen, testKitchen, logFolder, note)


    num_classes = 8
    bs = 1
    num_epochs = 101

    train_dataset = ek_train(shuffle = True, trainKitchen = trainKitchen)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, collate_fn=collate_fn2, drop_last = True)

    #dummy_x = torch.rand(bs, 3, 10, 224, 224).cuda()
    model = VIDEO_SWIN(num_classes = num_classes).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 40, 60, 80])

    #checkpoint = torch.load('./checkpoints/swin_base_patch244_window1677_sthv2.pth')
    #model.load_state_dict(checkpoint, strict=False)


  
    for epoch in range(num_epochs):
        train_epoch(model, train_dataloader, criterion, optimizer, logFile, epoch, True)
        lr_sched.step()
        if (epoch % 5 == 0):
            if (epoch > -1):
                print("ENTERING EVALUATION")
                train_accuracy(model,epoch, logFile, trainKitchen)
                validate(model,epoch, logFile, testKitchen)
                print()

    logFile.write("---------------------file close-------------------------------\n")
    logFile.close()


if __name__ == "__main__":
   main(sys.argv[1:])

