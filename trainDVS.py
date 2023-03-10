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
from dl_ft_1_train_O import ek_train, collate_fn2
from dl_ft_1_test_O import ek_test, collate_fn_test
from functions_NEK import train_accuracy, validate, train_epoch, getInputs, startLog

import sys
sys.path.append('/home/tr248228/RP_EvT/October/videoMae/snntorch/snntorch')
from spikevision.spikedata.dvs_gesture224 import DVSGesture



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
    logFile = startLog(trainKitchen, testKitchen, logFolder, note, "swin")

    num_classes = 8
    bs = 2
    use_shed = True
    eventDrop = True

    train_set = DVSGesture("D:/Downloads/DVS  Gesture dataset", train=True, num_steps=10, dt=50000, eventDrop = eventDrop)
    test_set = DVSGesture("D:/Downloads/DVS  Gesture dataset", train=False, num_steps=10, dt=180000,eventDrop = eventDrop)
    print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
    print(f'total samples = {train_set.__len__() + test_set.__len__()}')
    train_dataloader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=4, drop_last = True)

    #dummy_x = torch.rand(bs, 3, 10, 224, 224).cuda()
    model = VIDEO_SWIN(num_classes = num_classes).cuda()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)#, momentum=0.8, weight_decay=1e-6)
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-7)
    print("optimizer", optimizer)
    logFile.write("optimizer\n" + str(optimizer) + "\n")
    print("batch size", bs)
    logFile.write("batch_size\n" + str(bs) + "\n")
    if (use_shed):
        ms = [15, 30, 45]
        gm = 0.4
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=ms, gamma=gm)
        print("lr_sched", ms, gm)
        logFile.write("lr_sched\n" + str(ms) + ", " + str(gm) + "\n")
    else:
        print("no lr_sched")
    logFile.write("no lr_sched")
    
    steps = 0
    class_count = [164,679,242,210,119,39,1,113] 
    weights = 1 - (torch.tensor(class_count)/1567)
    weights = weights.cuda()
    criterion= torch.nn.CrossEntropyLoss(weight=weights).cuda() # new loss added
    model.train()



  
    for epoch in range(151):
        losses = []
        for i, data in enumerate(train_dataloader, 0):  
            #print('iter: ' + str(i))
            inputs, labels, pathBS = data
            optimizer.zero_grad()
            inputs = inputs.permute(0,2,1,3,4) #aug_DL output is [120, 16, 3, 112, 112]], #model expects [8, 3, 16, 112, 112]
            
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            labels = torch.as_tensor(labels)
            labels = Variable(labels.cuda())
            per_frame_logits = model(inputs).squeeze()
            if (i == 0) & (epoch == 0):
                print("inputs.shape", inputs.shape)
                print("per_frame_logits.shape", per_frame_logits.shape)
                print("labels.shape", labels.shape)
                print("labels.long().shape", labels.long().shape)      
            loss = criterion(per_frame_logits,labels.long())
            losses.append(loss.cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            steps += 1
            if (steps+1) % 250 == 0: 
                print('Epoch: {} Loss: {:.4f}'.format(epoch,np.mean(losses)))
        signal = "===============================================\n"
        eoe = "End of epoch " + str(epoch) + ", mean loss: " + str(np.mean(losses)) + "\n"
        print(signal + eoe + signal)
        logFile.write(signal + eoe + signal)
        logFile.flush()
        if (use_shed):
            lr_sched.step()
        if(epoch%4==0):
            train_accuracy(model,epoch, logFile, ek_test, collate_fn_test, 'p01')
            validate(model,epoch, logFile, ek_test, collate_fn_test, 'p01')
            validate(model,epoch, logFile, ek_test, collate_fn_test, 'p08')
            # torch.save(model.module.state_dict(), save_model+str(epoch).zfill(6)+'.pt')
            model.train()
            # now = datetime.now()
            # d8 = now.strftime("%d%m%Y")
            # current_time = now.strftime("%H:%M:%S")
            # weightStatus = d8 + " | " + current_time + " saving weights to: " + save_model +str(epoch)
            # print(weightStatus)
            # logFile.write(weightStatus + "\n")

    logFile.write("---------------------file close-------------------------------\n")
    logFile.close()


if __name__ == "__main__":
   main(sys.argv[1:])
