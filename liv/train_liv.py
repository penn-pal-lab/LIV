import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
from PIL import Image  
import clip 
import glob 
import hydra
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd 
import torch
import torchvision.transforms as T
import time

from liv import load_liv
from liv.trainer import Trainer
from liv.utils import utils
from liv.utils.data_loaders import LIVBuffer
from liv.utils.logger import Logger
from liv.utils.plotter import plot_reward_curves

def make_network(cfg):
    model =  hydra.utils.instantiate(cfg)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    if cfg.device == "cpu":
        model = model.module.to(cfg.device)
    return model

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logging = self.cfg.logging
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        if self.logging:
            self.setup()

        if not cfg.eval:
            print("Creating Dataloader")
            train_iterable = LIVBuffer(datasource=self.cfg.dataset, datapath=self.cfg.datapath_train, num_workers=self.cfg.num_workers, num_demos=self.cfg.num_demos, doaug=self.cfg.doaug, alpha=self.cfg.alpha)
            self.train_dataset = train_iterable
            self.train_loader = iter(torch.utils.data.DataLoader(train_iterable,
                                            batch_size=self.cfg.batch_size,
                                            num_workers=self.cfg.num_workers,
                                            pin_memory=True))

        ## Init Model
        print("Initializing Model")
        self.model = make_network(cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0

        ## If reloading existing model
        if cfg.load_snap:
            print("LOADING", cfg.load_snap)
            self.load_snapshot(cfg.load_snap)

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=False, cfg=self.cfg)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.train_steps, 1)
        eval_freq = self.cfg.eval_freq
        eval_every_step = utils.Every(eval_freq, 1)

        # trainer = Trainer()
        trainer = hydra.utils.instantiate(self.cfg.trainer)

        ## Training Loop
        print("Begin Training")
        while train_until_step(self.global_step):
            if eval_every_step(self.global_step):
                self.generate_reward_curves()
                self.save_snapshot()
            
            ## Sample Batch
            t0 = time.time()
            batch = next(self.train_loader)
            t1 = time.time()
            metrics, st = trainer.update(self.model, batch, self.global_step)
            t2 = time.time()
            if self.logging:
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            if self.global_step % 10 == 0:
                print(self.global_step, metrics)
                print(f'Sample time {t1-t0}, Update time {t2-t1}')
                
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.global_step}.pt'
        global_snapshot =  self.work_dir / f'snapshot.pt'
        sdict = {}
        sdict["liv"] = self.model.module.state_dict()
        sdict["optimizer"] = self.model.module.encoder_opt.state_dict()
        sdict["global_step"] = self._global_step
        torch.save(sdict, snapshot)
        torch.save(sdict, global_snapshot)

    def load_snapshot(self, snapshot_path):
        if snapshot_path != 'liv':
            payload = torch.load(snapshot_path)
            self.model.module.load_state_dict(payload['liv'])
        else:
            self.model = load_liv()
        clip.model.convert_weights(self.model)
        try:
            self._global_step = payload['global_step']
        except:
            print("Warning: No global step found")

    def generate_reward_curves(self):
        self.model.eval()
        os.makedirs(f"{self.work_dir}/reward_curves", exist_ok=True)
        transform = T.Compose([T.ToTensor()])

        if self.cfg.dataset not in ["epickitchen"]:
            manifest = pd.read_csv(os.path.join(self.cfg.datapath_train, "manifest.csv"))
            tasks = manifest["text"].unique()
        else:
            manifest = pd.read_csv(os.path.join(self.cfg.datapath_train, "EPIC_100_validation.csv"))
            tasks = ["open microwave", "open cabinet", "open door"]

        fig_filename = f"{self.work_dir}/reward_curves/{self._global_step}_{self.cfg.dataset}"
        if self.cfg.dataset in ["epickitchen"]:
            def load_video(m):
                imgs_tensor = []
                start_frame = m["start_frame"]
                end_frame = m["stop_frame"]
                vid = f"/data2/jasonyma/EPIC-KITCHENS/frames/{m['participant_id']}/rgb_frames/{m['video_id']}"
                for index in range(start_frame, end_frame):
                    img = Image.open(f"{vid}/frame_0000{index+1:06}.jpg")
                    imgs_tensor.append(transform(img))
                imgs_tensor = torch.stack(imgs_tensor)
                return imgs_tensor

        else:
            def load_video(m):
                imgs_tensor = []
                vid = m["directory"]
                for index in range(m["num_frames"]):
                    try:
                        img = Image.open(f"{vid}/{index}.png")
                    except:
                        img = Image.open(f"{vid}/{index}.jpg")
                    imgs_tensor.append(transform(img))
                imgs_tensor = torch.stack(imgs_tensor)
                return imgs_tensor

        plot_reward_curves(
            manifest,
            tasks,
            load_video,
            self.model,
            fig_filename,
            animated=self.cfg.animate,
        )
        self.model.train()


@hydra.main(config_path='cfgs', config_name='config_liv')
def main(cfg):
    from liv.train_liv import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)

    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot(snapshot)

    if not cfg.eval:
        workspace.train()
    else:
        workspace.generate_reward_curves()

if __name__ == '__main__':
    main()