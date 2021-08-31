import torch
import torch.nn as nn

from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from utils.log import log


class BaseGAN:
    def __init__(self, cfg, writer):
        """
        __init__ BaseGAN consturctor

        Args:
            cfg (Munch): Munch Config Object
        """
        super().__init__()
        self.cfg = cfg
        self.n_critic = 1 if self.cfg.n_critic is None else self.cfg.n_critic

        self.writer = writer
        self.train_step = 0

        self.__post_epoch_hooks = []

    def _update_model_optimizers(self):
        pass

    def generator_step(self, data):
        raise NotImplementedError("method not implemented!")

    def critic_step(self, data):
        raise NotImplementedError("method not implemented!")

    def train_epoch(self, dataloader):
        self.metrics = defaultdict(list)

        loop = tqdm(dataloader, desc="Trg Itr: ", ncols=75, leave=False)

        for ix, data in enumerate(loop):
            self.critic_step(data)
            if ix % self.n_critic == 0:
                self.generator_step(data)

            self.log_step_losses(self.metrics, self.train_step)

            if ix % self.cfg.viz_freq == 0:
                self.vizualize_gen(dataloader, self.train_step)

            self.train_step += 1

        # * call the registered hooks
        for hook in self.__post_epoch_hooks:
            hook()

    def _generator_loss(self, **kwargs):
        pass

    def _critic_loss(self, **kwargs):
        pass

    def sample_noise(self):
        return torch.randn(self.cfg.batch_size, self.cfg.z_dim, 1, 1).to(
            self.cfg.device
        )

    def log_step_losses(self, metrics, step, train=True):
        ses = "train" if train else "val"
        for key in metrics:
            self.writer.add_scalar(
                f"{ses}-step/{key}", metrics[key][-1], step
            )

    def log_epoch_losses(self, metrics, step, train=True):
        ses = "train" if train else "val"
        for key in metrics:
            self.writer.add_scalar(
                f"{ses}-ep/{key}", np.mean(metrics[key]), step
            )

    def vizualize_gen(self, dataloader, step, n_samples=16):
        n_samples = min(n_samples, self.cfg.batch_size)
        fake_images = self.generate_images(n_samples=n_samples)
        real_images = iter(dataloader).next()[0][:n_samples]

        grid = make_grid(fake_images, nrow=4, normalize=True)
        self.writer.add_image("fake-images", grid, step)

        grid = make_grid(real_images, nrow=4, normalize=True)
        self.writer.add_image("real-images", grid, step)

        return

    def generate_images(self, n_samples):
        self.netG.eval()
        with torch.no_grad():
            noise = self.sample_noise()[:n_samples]
            fake_images = self.netG(noise)
        return fake_images

    def save_model(self, ckpt_dir: str, current_ep: int):
        out_path = os.path.join(ckpt_dir, f"netG-{(current_ep+1):03d}.tar")
        self._ckpt(self.netG, out_path)

        out_path = os.path.join(ckpt_dir, f"netD-{(current_ep+1):03d}.tar")
        self._ckpt(self.netD, out_path)

    def _ckpt(self, model, path):
        """
        _ckpt makes checkpoint

        Args:
            model (nn.Module): module to save
            path (str): save path
        """
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), path)
