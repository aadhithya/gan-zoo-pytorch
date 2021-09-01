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
        self.netD = NetD(
            self.cfg.img_ch,
            norm_layer=nn.BatchNorm2d,
            final_activation=torch.sigmoid,
        )

        self.netG.apply(init_weight)
        self.netD.apply(init_weight)

        # * DCGAN Alternating optimization
        self.n_critic = 1

        self.bce_loss = nn.BCELoss()

        self._update_model_optimizers()

    def _update_model_optimizers(self):
        self.netG = self.netG.to(self.cfg.device)
        self.netD = self.netD.to(self.cfg.device)

        self.optG = Adam(
            self.netG.parameters(), lr=self.cfg.lr.g, betas=(0.5, 0.999)
        )
        self.optD = Adam(
            self.netD.parameters(), lr=self.cfg.lr.d, betas=(0.5, 0.999)
        )

        self.netG.train()
        self.netD.train()

    def generator_step(self, data):
        self.optG.zero_grad()

        # noise = self.sample_noise()

        fake_logits = self.netD(self.fake_images)

        loss = self.bce_loss(fake_logits, torch.ones_like(fake_logits))

        loss.backward()
        self.optG.step()

        self.metrics["gen-loss"] += [loss.item()]

    def critic_step(self, data):
        self.optD.zero_grad()

        real_images = data[0].float().to(self.cfg.device)

        real_logits = self.netD(real_images).view(-1)
        real_loss = self.bce_loss(real_logits, torch.ones_like(real_logits))

        noise = self.sample_noise()
        self.fake_images = self.netG(noise)

        fake_logits = self.netD(self.fake_images).view(-1)
        fake_loss = self.bce_loss(fake_logits, torch.zeros_like(fake_logits))

        loss = real_loss + fake_loss

        loss.backward(retain_graph=True)
        self.optD.step()

        self.metrics["discriminator-loss"] += [loss.item()]
