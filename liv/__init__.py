import os 
from os.path import expanduser
import omegaconf
import hydra
from huggingface_hub import hf_hub_download
import gdown
import torch
import copy
from liv.models.model_liv import LIV

VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "num_negatives"]
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def cleanup_config(cfg):
    config = copy.deepcopy(cfg)
    config["device"] = device

    return config.agent

def load_liv(modelid='resnet50'):
    assert modelid == 'resnet50'
    home = os.path.join(expanduser("~"), ".liv")

    if not os.path.exists(os.path.join(home, modelid)):
        os.makedirs(os.path.join(home, modelid))
    folderpath = os.path.join(home, modelid)
    modelpath = os.path.join(home, modelid, "model.pt")
    configpath = os.path.join(home, modelid, "config.yaml")

    if not os.path.exists(modelpath):
        try:
            # Default reliable download from HuggingFace Hub
            hf_hub_download(repo_id="jasonyma/LIV", filename="model.pt", local_dir=folderpath)
            hf_hub_download(repo_id="jasonyma/LIV", filename="config.yaml", local_dir=folderpath)
        except:
            # Download from GDown
            modelurl = 'https://drive.google.com/uc?id=1l1ufzVLxpE5BK7JY6ZnVBljVzmK5c4P3'
            configurl = 'https://drive.google.com/uc?id=1GWA5oSJDuHGB2WEdyZZmkro83FNmtaWl'
            gdown.download(modelurl, modelpath, quiet=False)
            gdown.download(configurl, configpath, quiet=False)

    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    rep = torch.nn.DataParallel(rep)
    state_dict = torch.load(modelpath, map_location=torch.device(device))['liv']
    rep.load_state_dict(state_dict)
    return rep    

