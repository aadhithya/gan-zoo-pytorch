from base.model import BaseGAN
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as tf

import os
import yaml
from munch import DefaultMunch
from tqdm import trange
import random
import numpy as np
from tensorboardX import SummaryWriter

from utils.log import log


class Trainer:
    def __init__(self, Model: BaseGAN, cfg_path) -> None:
        super().__init__()

        self.__init_config(cfg_path)

        self.__set_seed()

        self.__init_writer()

        transforms = tf.Compose(
            [
                tf.ToTensor(),
                tf.Normalize(0.5, 0.5),
                tf.Resize((self.cfg.imsize, self.cfg.imsize)),
            ]
        )

        self.dataset = ImageFolder(
            root=self.cfg.data_dir, transform=transforms
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        self.model = Model(self.cfg, self.writer)

        self.current_ep = 0

    def __init_config(self, cfg_path):
        log.info(f"Loading config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = DefaultMunch.fromDict(cfg)

    def __generate_run_id(self):
        run_id = 0
        if os.path.exists("./.runid"):
            with open("./.runid", "r") as f:
                run_id = int(f.read()) + 1
        with open("./.runid", "w") as f:
            f.write(str(run_id))

        return run_id

    def __init_writer(self):
        run_id = self.__generate_run_id()
        self.log_dir = f"./.temp/{run_id}/"
        self.ckpt_dir = os.path.join(self.log_dir, "ckpt")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

    def __set_seed(self):
        torch.backends.cudnn.deterministic = True
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        log.info("Seed Set...🥜")

    def train(self):
        loop = trange(self.cfg.epochs, desc="Epoch: ", ncols=75)
        for ep in enumerate(loop):
            self.model.train_epoch(self.dataloader)
            self.current_ep += 1
