import clip
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from pathlib import Path
from torchvision.utils import save_image
import time
import copy
import torchvision.transforms as T

epsilon = 1e-8
def do_nothing(x): return x

class Trainer():
    def __init__(self):
        self.clip_loss_img = nn.CrossEntropyLoss()
        self.clip_loss_txt = nn.CrossEntropyLoss()
    
    # A simplified function call of CLIP.forward()
    def compute_clip_loss(self, model, image_features, text_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        ground_truth = torch.arange(len(image_features),dtype=torch.long,device=image_features.device)
        clip_loss = (self.clip_loss_img(logits_per_image,ground_truth) + self.clip_loss_txt(logits_per_text,ground_truth))/2
        return clip_loss 

    def compute_vip_loss(self, model, e0, es0_vip, es1_vip, eg, b_reward, num_negatives=0):
        r =  b_reward.to(e0.device)

        V_0 = model.module.sim(e0, eg)
        V_s = model.module.sim(es0_vip, eg)
        V_s_next = model.module.sim(es1_vip, eg)

        # Rescale Value 
        V_0 = V_0 / (1-model.module.gamma)
        V_s = V_s / (1-model.module.gamma)
        V_s_next = V_s_next / (1-model.module.gamma)

        # Compute VIP Loss
        V_loss = (1-model.module.gamma) * -V_0.mean() + torch.log(epsilon + torch.mean(torch.exp(-(r + model.module.gamma * V_s_next - V_s))))

        # Optionally, add additional "negative" observations
        if num_negatives > 0:
            V_s_neg = []
            V_s_next_neg = []
            for _ in range(num_negatives):
                perm = torch.randperm(es0_vip.size()[0])
                es0_vip_shuf = es0_vip[perm]
                es1_vip_shuf = es1_vip[perm]

                V_s_neg.append(model.module.sim(es0_vip_shuf, eg))
                V_s_next_neg.append(model.module.sim(es1_vip_shuf, eg))

            V_s_neg = torch.cat(V_s_neg)
            V_s_next_neg = torch.cat(V_s_next_neg)
            r_neg = -torch.ones(V_s_neg.shape).to(V_0.device)
            V_s_neg = V_s_neg / (1-model.module.gamma)
            V_s_next_neg = V_s_next_neg / (1-model.module.gamma)                
            V_loss = V_loss + torch.log(epsilon + torch.mean(torch.exp(-(r_neg + model.module.gamma * V_s_next_neg - V_s_neg))))
        
        return V_loss 

    def update(self, model, batch, step, eval=False):
        t0 = time.time()
        metrics = dict()
        if eval:
            model.eval()
        else:
            model.train()

        ## Batch
        b_im, b_reward, b_lang = batch
        b_im = b_im.cuda()
        bs = b_im.shape[0]
        img_stack_size = b_im.shape[1]
        H = b_im.shape[-2]
        W = b_im.shape[-1]
        b_im_r = b_im.reshape(bs*img_stack_size, 3, H, W)

        # Encode visual and text inputs
        e_img = model(b_im_r, modality="vision")
        b_token = clip.tokenize(b_lang)
        e_lang = model(b_token, modality="text")

        e_img = e_img.reshape(bs, img_stack_size, -1)
        e0 = e_img[:, 0] # initial, o_0
        eg = e_img[:, 1] # final, o_g
        es0_vip = e_img[:, 2] # o_t
        es1_vip = e_img[:, 3] # o_t+1
        eg_img = e_img[:, -1] 
        full_loss = 0

        ## CLIP Loss 
        if model.module.clipweight != 0:
            clip_loss = self.compute_clip_loss(model, eg_img, e_lang)
            clip_loss = model.module.clipweight * clip_loss
            metrics['clip_loss'] = clip_loss.item()
            full_loss += clip_loss

        ## VIP Loss (Visual)
        vip_loss_visual = self.compute_vip_loss(model, e0, es0_vip, es1_vip, eg, b_reward, model.module.num_negatives)
        metrics['vip_loss_visual'] = vip_loss_visual.item()
        full_loss += model.module.visionweight * vip_loss_visual

        ## VIP Loss (Language)
        if model.module.langweight != 0:
            vip_loss_lang = self.compute_vip_loss(model, e0, es0_vip, es1_vip, e_lang, b_reward, model.module.num_negatives)
            metrics['vip_loss_lang'] = vip_loss_lang.item()
            full_loss += model.module.langweight * vip_loss_lang

        metrics['full_loss'] = full_loss.item()

        if not eval:
            model.module.encoder_opt.zero_grad()
            full_loss.backward()
            model.module.encoder_opt.step()

        return metrics, None
