import torch
from torchvision.datasets import CelebA
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tf

import yaml
from munch import DefaultMunch
from tqdm import trange

class Trainer:
    def __init__(self, cfg_path="./config/config.yml") -> None:
        super().__init__()

        self.__init_config(cfg_path)

        transforms = tf.Compose([
            tf.ToTensor(), tf.Normalize(0.5, 0.5),
            tf.Resize(self.cfg.imsize)
        ])

        self.dataset = CelebA("./.temp", transform=transforms, download=self.cfg.download)
        self.dataloader = DataLoader(self.dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=2)

        self.current_ep = 0

    def __init_config(self, cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = DefaultMunch.fromDict(cfg)
    
    def train(self):
        loop = trange(
            self.cfg.epochs, desc="Epoch: ",
            ncols=75
        )
        for ep in enumerate(loop):
            self.model.train_epoch(self.dataloader)
            self.current_ep += 1