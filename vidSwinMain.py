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
import os
from datetime import datetime

import torch
import torch.nn as nn
from video_swin_transformer import SwinTransformer3D
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader

sys.path.append('/home/tr248228/RP_EvT/October/n-epic-kitchens')
from dl_ft_1_train_O import ek_train, collate_fn2
from dl_ft_1_test_O import ek_test, collate_fn_test
from functions_NEK import train_accuracy, validate, getInputs, startLog, validateDVS
sys.path.append('/home/tr248228/RP_EvT/October/videoMae/snntorch/snntorch')
from spikevision.spikedata.dvs_gesture224 import DVSGesture


class VIDEO_SWIN(nn.Module):
    def __init__(self, num_classes = 8, feature_size = 1024, pretrained = True):
        super(VIDEO_SWIN, self).__init__()
        #self.backbone  = SwinTransformer3D()

        self.backbone = SwinTransformer3D(embed_dim=128, 
                          depths=[2, 2, 18, 2], 
                          num_heads=[4, 8, 16, 32], 
                          patch_size=(2,4,4), 
                          window_size=(16,7,7), 
                          drop_path_rate=0.4, 
                          patch_norm=True)
        if pretrained:
            checkpoint = torch.load('./checkpoints/swin_base_patch244_window1677_sthv2.pth')

            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                if 'backbone' in k:
                    name = k[9:]
                    new_state_dict[name] = v 

            self.backbone.load_state_dict(new_state_dict)
        else:
            print("________NOT PRETRAINED________")
        self.class_head = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool3d(self.backbone(x),1).flatten(1)
        x = self.class_head(x)
        return x




def main(argv):
    torch.cuda.empty_cache()
    batch_size=4
    num_epochs = 100
    use_shed = False
    cosinelr = False
    original = True
    eventDrop = True
    randomcrop = False
    dataset = 'DVS' # 'NEK' 'DVS'
    three_layer_frozen = False
    two_layer_frozen = False
    changing_sr  = False
    rdCrop_fr  = False
    if rdCrop_fr:
        randomcrop = True
    evtDropPol = False
    pretrained = False

    evAugs = ["rand"]
    num_steps = 100
    final_frames = 16
    skip_rate = 5
    assert final_frames*skip_rate <= num_steps

    trainKitchen, testKitchen, logFolder, note = getInputs(argv)
    logFile = startLog(trainKitchen, testKitchen, logFolder, note, "swin")
    save_model='./saved_models/' + logFolder + '/'
    if not os.path.exists(save_model):
        os.makedirs(save_model)
    num_classes = 11

    print(dataset)
    logFile.write(dataset + "\n")
    print("eventDrop", eventDrop)
    logFile.write("eventDrop" + str(eventDrop) + "\n")

    if dataset == 'NEK':
        #evAugs = ["val", "rand", "time", "rect", "pol"]
        evAugs = ["val", "rand", "time", "rect", "pol"]
        print("evAugs", evAugs)
        logFile.write("evAugs" + str(evAugs) + "\n")
        train_dataset = ek_train(shuffle = True, trainKitchen = 'p01', eventDrop = eventDrop, eventAugs = evAugs)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                        collate_fn=collate_fn2, drop_last = True)
    elif dataset == 'DVS':
        
        train_set = DVSGesture("/home/tr248228/RP_EvT/October/videoMae/DVS/download", train=True, dt = int(500000/num_steps), num_steps=num_steps, eventDrop = eventDrop, eventAugs = evAugs, skip_rate = skip_rate, final_frames=final_frames, randomcrop = randomcrop, changing_sr = changing_sr, rdCrop_fr = rdCrop_fr, evtDropPol = evtDropPol)
        print(f'train samples = {train_set.__len__()}')
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, drop_last = True)

        print("num_steps", num_steps)
        logFile.write("num_steps" + str(num_steps) + "\n")
        print("final_frames", final_frames)
        logFile.write("final_frames" + str(final_frames) + "\n")
        print("skip_rate", skip_rate)
        logFile.write("skip_rate" + str(skip_rate) + "\n")
    print("eventDrop", eventDrop)
    logFile.write("eventDrop" + str(eventDrop) + "\n")
    print("evAugs", evAugs)
    logFile.write("evAugs" + str(evAugs) + "\n")
    print("randomcrop", randomcrop)
    logFile.write("randomcrop" + str(randomcrop) + "\n")
    print("pretrained", pretrained)
    logFile.write("pretrained" + str(pretrained) + "\n")

    model = VIDEO_SWIN(num_classes = num_classes, pretrained = pretrained).cuda()
    if torch.cuda.device_count()>1:
        print(f'Multiple GPUS found!')
        model=nn.DataParallel(model)
        model.cuda()
        
    else:
        print('Only 1 GPU is available')
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    print("optimizer", optimizer)
    logFile.write("optimizer\n" + str(optimizer) + "\n")
    print("batch size", batch_size)
    logFile.write("batch_size\n" + str(batch_size) + "\n")
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
    if dataset == "NEK":
        class_count = [164,679,242,210,119,39,1,113]
        weights = 1 - (torch.tensor(class_count)/1567)
        weights = weights.cuda()
        criterion= torch.nn.CrossEntropyLoss(weight=weights.float()).cuda()
    elif dataset == "DVS":
        criterion= torch.nn.CrossEntropyLoss().cuda()
    #new loss added
    model.train()



    bestacc = -1
    acc = 0
    for epoch in range(num_epochs + 1):
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
            if (dataset == "NEK"):
                train_accuracy(model,epoch, logFile, ek_test, collate_fn_test, 'p01')
                validate(model,epoch, logFile, ek_test, collate_fn_test, 'p01')
                validate(model,epoch, logFile, ek_test, collate_fn_test, 'p08')
            elif dataset == 'DVS':
                acc = validateDVS(model,epoch, logFile, DVSGesture, "swin", num_steps, final_frames, skip_rate)
                if acc > bestacc:
                    bestacc = acc
                    logFile.write("BEST!!!! \n")
            # torch.save(model.module.state_dict(), save_model+str(epoch).zfill(6)+'.pt')
            model.train()
            torch.save(model.state_dict(), save_model+str(epoch).zfill(6)+'.pt')
            now = datetime.now()
            d8 = now.strftime("%d%m%Y")
            current_time = now.strftime("%H:%M:%S")
            weightStatus = d8 + " | " + current_time + " saving weights to: " + save_model +str(epoch)
            print(weightStatus)
            logFile.write(weightStatus + "\n")
    print("bestacc: ", bestacc)
    logFile.write("bestacc: " + str(bestacc) + "\n")
    logFile.write("---------------------file close-------------------------------\n")
    logFile.close()

# def validateDVS(model,epoch, logFile, eventDrop):
#   print(f'*************************Training Accuracy********************')
#   print(f'Checking Training Accuracy at epoch {epoch}')
#   model.eval()
#   bs = 1

#   test_set = DVSGesture("/home/tr248228/RP_EvT/October/videoMae/DVS/download", train=False, num_steps=10, dt=180000, eventDrop = eventDrop)
#   test_dataloader = DataLoader(test_set, batch_size=bs, shuffle=True, num_workers=0, drop_last = True)

#   criterion = torch.nn.CrossEntropyLoss()
#   optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#   losses = []
#   for i, data in enumerate(test_dataloader, 0):
#     video, clip_label = data
#     # video = np.einsum('ijklm->ijlmk',video)
#     video = video.permute(0,2,1,3,4) # new
#     video = Variable(video.cuda()) 
#     clip_label = Variable(clip_label.cuda())
#     # clip_label = torch.as_tensor(clip_label).cuda()
#     # inputs = feature_extractor(list(video), return_tensors="pt")
#     pred = model(video).logits.squeeze()
#     sftmx = torch.nn.Softmax(dim=0)
#     pred_clip_1 = sftmx(pred)
#     pred_clip_1 = torch.tensor([torch.argmax(pred_clip_1)]).cuda()
#     losses.append(int(pred_clip_1 == clip_label))
#   acc = np.mean(losses)
#   print("test accuracy: " + str(acc*100) + "\n")
#   print(f'**************************************************************')
#   logFile.write("test accuracy: " + str(acc*100) + "\n")


if __name__ == "__main__":
   main(sys.argv[1:])

