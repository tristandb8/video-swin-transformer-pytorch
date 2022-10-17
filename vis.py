from bdb import Breakpoint
import sys, getopt
from datetime import datetime
import time
import cv2
import numpy as np

from torch.utils.data import DataLoader
from dl_ft_1_train import ek_train, collate_fn2
from dl_ft_1_test import ek_test, collate_fn_test

def vis_frames(clip,name,path):
  temp = clip[0,:]
  temp = clip.permute(2,3,1,0)
 
  frame_width = 224
  frame_height = 224
  frame_size = (frame_width,frame_height)
  path = path + '/' +  name + '.avi'
  video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc('p', 'n', 'g', ' '),2,(frame_size[1],frame_size[0]))
  
  for i in range(temp.shape[3]):
    x = np.array(temp[:,:,:,i])
    x *= 255/(x.max()) 
    x[x>255] = 255
    x[x<0] = 0
    x = x.astype(np.uint8)
    #x = np.clip(x, a_min = -0.5, a_max = 0.5)
    video.write(x) 
  video.release()

def main(argv):
    actions = ['put','take','open','close','wash','cut','mix','pour']
    train_dataset = ek_train(shuffle = True, trainKitchen = 'p01')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn2, drop_last = True)
    for iter, (clip, clip_label,vid_path) in enumerate(train_dataloader):
        print(clip_label)
        print(actions[clip_label[0].item()])
        vis_frames(clip[0], actions[clip_label[0].item()] + str(iter), 'clips')
        if (iter > 39):
            break
    
    print("done")

if __name__ == "__main__":
   main(sys.argv[1:])