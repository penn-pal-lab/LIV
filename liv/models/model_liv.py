import clip
from clip.model import CLIP
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T

class LIV(nn.Module):
    def __init__(self, modelid="RN50", device="cuda",
                       lr=1e-5, weight_decay=0.001,
                       visionweight=1.0, langweight=1.0, clipweight=1.0,
                       gamma=0.98, metric="cos", num_negatives=0,
                       grad_text=True, scratch=False):
        super().__init__()

        self.modelid = modelid
        self.device = device
        self.visionweight = visionweight 
        self.langweight = langweight
        self.clipweight = clipweight

        self.gamma = gamma
        self.num_negatives = num_negatives
        self.metric = metric
        self.grad_text = grad_text

        # Load CLIP model and transform
        model, cliptransforms = clip.load(modelid, device=self.device, scratch=scratch, jit=False)
        
        # CLIP precision
        if device == "cpu":
            model.float()
        else :
            clip.model.convert_weights(model)
        
        self.model = model 
        self.model.train()
        self.transforms = cliptransforms

        self.transforms_tensor = nn.Sequential(
                transforms.Resize(self.model.visual.input_resolution, antialias=None),
                transforms.CenterCrop(self.model.visual.input_resolution),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            )

        self.output_dim = self.model.visual.output_dim

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(list(self.model.parameters()),
        lr=lr, betas=(0.9,0.98),eps=1e-6, weight_decay=weight_decay)      

    ## Forward Call (im --> representation)
    def forward(self, input, modality="vision", normalize=True):
        if modality == "vision":
            if type(input) != torch.Tensor:
                print("Warning: Input not tensor, may cause issue with normalization")
                input = self.transforms(input).to(self.device)
            else:
                # rescale to [0, 1]
                if torch.max(input) > 10.0:
                    input = input / 255.0
                input = self.transforms_tensor(input).to(self.device)

            features = self.model.encode_image(input)
        elif modality == "text":
            b_token = input
            if self.grad_text:
                features = self.model.encode_text(b_token)
            else:
                with torch.no_grad():
                    features = self.model.encode_text(b_token)
        else:
            raise NotImplementedError

        return features

    def sim(self, tensor1, tensor2):
        if type(tensor1) == np.ndarray:
            tensor1 = torch.from_numpy(tensor1).to(self.device)
            tensor2 = torch.from_numpy(tensor2).to(self.device)
        if self.metric == 'l2':
            d = -torch.linalg.norm(tensor1 - tensor2, dim = -1)
        elif self.metric == 'cos':
            tensor1 = tensor1 / tensor1.norm(dim=-1, keepdim=True)
            tensor2 = tensor2 / tensor2.norm(dim=-1, keepdim=True)
            d = torch.nn.CosineSimilarity(-1)(tensor1, tensor2)
        else:
            raise NotImplementedError
        return d
        
    def get_reward(self, e0, es, le, encoded=True):
        assert encoded == True 
        return self.sim(es, le)
    
