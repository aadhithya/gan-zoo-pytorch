import torch
import torch.nn as nn
from torch import autograd

import os

from torch.optim.adam import Adam

from base.model import BaseGAN
from models.modules.net import NetG, NetD
from utils.utils import init_weight


class DCGAN(BaseGAN):
    """
    Deep Convolutional GAN https://arxiv.org/pdf/1511.06434.pdf
    """

    def __init__(self, cfg, writer):
        super().__init__(cfg, writer)

        self.netG = NetG(
            z_dim=self.cfg.z_dim,
            out_ch=self.cfg.img_ch,
            norm_layer=nn.BatchNorm2d,
            final_activation=torch.tanh,
        )
        self.netD = NetD(self.cfg.img_ch, norm_layer=nn.BatchNorm2d)

        # * DCGAN Alternating optimization
        self.n_critic = 1

        self.bce_loss = nn.BCEWithLogitsLoss()

        self._update_model_optimizers()

    def _update_model_optimizers(self):
        self.netG = self.netG.to(self.cfg.device)
        self.netD = self.netD.to(self.cfg.device)

        self.optG = Adam(self.netG.parameters(), lr=self.cfg.lr.g)
        self.optD = Adam(self.netD.parameters(), lr=self.cfg.lr.d)

    def generator_step(self, data):
        self.netG.train()
        self.netD.eval()

        self.optG.zero_grad()

        noise = self.sample_noise()

        fake_images = self.netG(noise)

        fake_logits = self.netD(fake_images)

        loss = self.bce_loss(fake_logits, torch.ones_like(fake_logits))

        loss.backward()
        self.optG.step()

        self.metrics["gen-loss"] += [loss.item()]

    def critic_step(self, data):
        self.netG.eval()
        self.netD.train()

        self.optD.zero_grad()

        real_images = data[0].float().to(self.cfg.device)

        real_logits = self.netD(real_images)
        real_loss = self.bce_loss(real_logits, torch.ones_like(real_logits))
        real_loss.backward()

        noise = self.sample_noise()
        fake_images = self.netG(noise)

        fake_logits = self.netD(fake_images)
        fake_loss = self.bce_loss(fake_logits, torch.zeros_like(fake_logits))
        fake_loss.backward()

        loss = real_loss + fake_loss

        self.optD.step()

        self.metrics["discriminator-loss"] += [loss.item()]
