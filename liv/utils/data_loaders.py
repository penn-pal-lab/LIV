import warnings

import torchvision
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
from pathlib import Path
import glob
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import re
import random

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    return l 

def get_ind(vid, index, ds="ego4d"):
    if "epickitchen" in ds:
        return torchvision.io.read_image(f"{vid}/frame_0000{index+1:06}.jpg")
    else:
        try:
            return torchvision.io.read_image(os.path.join(vid, f"{index}.jpg"))
        except: 
            return torchvision.io.read_image(os.path.join(vid, f"{index}.png"))

## Data Loader for LIV
class LIVBuffer(IterableDataset):
    def __init__(self, datasource='epickitchen', datapath=None, num_workers=8, num_demos=100, doaug = "none", alpha=0.95):
        self._num_workers = max(1, num_workers)
        self.num_demos = num_demos
        self.alpha = alpha
        self.datasource = datasource
        self.datapath = datapath
        assert(datapath is not None)
        self.doaug = doaug
        
        # augmentations
        self.preprocess = torch.nn.Sequential(
                        transforms.Resize(256, antialias=None),
                        transforms.CenterCrop(224)
                )

        # default data augmentation 
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale = (0.2, 1.0), antialias=None),
            )
        elif doaug in ["metaworld"]:
            self.aug = torch.nn.Sequential(
                transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
                transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            )
        else:
            self.aug = lambda a : a

        # Load data
        if "epickitchen" in self.datasource: 
            self.manifest = pd.read_csv(f"{self.datapath}/EPIC_100_train.csv")
            self.datalen = len(self.manifest)
        else:
            self.manifest = pd.read_csv(f"{self.datapath}/manifest.csv")
            self.datalen = len(self.manifest)

    def _sample(self):
        # Sample a video from datasource
        if self.datasource == 'epickitchen':
            vidid = np.random.randint(0, self.datalen)
            m = self.manifest.iloc[vidid]
            start_frame = m["start_frame"]
            end_frame = m["stop_frame"]
            vidlen = end_frame - start_frame + 1
            text = m["narration"]
            vid = f"/data2/jasonyma/EPIC-KITCHENS/frames/{m['participant_id']}/rgb_frames/{m['video_id']}"
        else: 
            vidid = np.random.randint(0, self.datalen)
            m = self.manifest.iloc[vidid]
            vidlen = m["num_frames"]
            text = m["text"]
            vid = m["directory"]
            start_frame = 0
            end_frame = vidlen

        # Sample (o_t, o_k, o_k+1, o_T) for LIV training
        if self.datasource in ["epickitchen"]:
            start_ind = np.random.randint(0, vidlen-2)
            end_ind = np.random.randint(start_ind+1, vidlen)
            end_text_ind = np.random.randint(int(self.alpha * vidlen) - 1, vidlen)
        else:
            start_ind = 0
            end_ind = vidlen - 1
            end_text_ind = vidlen - 1

        start_ind = start_ind + start_frame 
        end_ind = end_ind + start_frame 
        end_text_ind = end_text_ind + start_frame 

        s0_ind = np.random.randint(start_ind, end_ind)
        s1_ind = min(s0_ind+1, end_ind)

        # Self-supervised reward (this is always -1)
        reward = float(s0_ind == end_ind) - 1 

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, self.datasource) 
            img = get_ind(vid, end_ind, self.datasource)
            img_text = get_ind(vid, end_text_ind, self.datasource)
            imts0 = get_ind(vid, s0_ind, self.datasource)
            imts1 = get_ind(vid, s1_ind, self.datasource)

            allims = torch.stack([im0, img, imts0, imts1, img_text], 0)
            allims_aug = self.aug(allims / 255.0) 

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            img_text = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, self.datasource) / 255.0) 
            img = self.aug(get_ind(vid, end_ind, self.datasource) / 255.0)
            imts0 = self.aug(get_ind(vid, s0_ind, self.datasource) / 255.0) 
            imts1 = self.aug(get_ind(vid, s1_ind, self.datasource) / 255.0) 
            img_text = self.aug(get_ind(vid, end_text_ind, self.datasource) / 255.0) 

        im = torch.stack([im0, img, imts0, imts1, img_text])
        return (im, reward, text)

    def __iter__(self):
        while True:
            yield self._sample()

if __name__ == "__main__":
    # You can test your dataset here by replacing the dummy datapath 
    datasource = ""
    datapath = ""
    buffer = LIVBuffer(datasource=datasource, datapath=datapath, doaug='rctraj')
    train_loader = iter(torch.utils.data.DataLoader(buffer, batch_size=2, num_workers=1, pin_memory=True))
    for i in range(1):
        img, reward, text, action = next(train_loader)
        print(img.shape, reward.shape, action.shape, text)