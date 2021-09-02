import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch import autograd

import os

from base.model import BaseGAN
from models.modules.net import NetG, NetD
from utils.utils import init_weight


class WGAN(BaseGAN):
    """
    WGAN Wasserstein Generative Adversarial Network
    Uses Wasserstein distance for training GAN and
    gradient clipping to enforce 1-Lipschitz continuity.
    http://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf
    """

    def __init__(self, cfg, writer):
        super().__init__(cfg, writer)

        self.netG = NetG(
            z_dim=self.cfg.z_dim,
            out_ch=self.cfg.img_ch,
            norm_layer=nn.BatchNorm2d,
            final_activation=torch.tanh,
        )
        self.netD = NetD(self.cfg.img_ch, norm_layer=nn.InstanceNorm2d)
        self.netG.apply(init_weight)
        self.netD.apply(init_weight)

        self.n_critic = self.cfg.n_critic
        self.c = 0.01 if self.cfg.c is None else self.cfg.c

        self._update_model_optimizers()

    def _update_model_optimizers(self):
        self.netG = self.netG.to(self.cfg.device)
        self.netD = self.netD.to(self.cfg.device)

        self.optG = RMSprop(self.netG.parameters(), lr=self.cfg.lr.g)
        self.optD = RMSprop(self.netD.parameters(), lr=self.cfg.lr.d)

    def generator_step(self, data):
        self.netG.train()
        self.netD.eval()

        self.optG.zero_grad()

        noise = self.sample_noise()

        fake_images = self.netG(noise)

        fake_logits = self.netD(fake_images)
       
        # * min E_{x~P_X}[C(x)] - E_{Z~P_Z}[C(g(z))]
        loss = -fake_logits.mean().view(-1)

        loss.backward()
        self.optG.step()

        self.metrics["gen-loss"] += [loss.item()]

    def critic_step(self, data):
        self.netG.eval()
        self.netD.train()

        self.optD.zero_grad()

        real_images = data[0].float().to(self.cfg.device)

        noise = self.sample_noise()
        fake_images = self.netG(noise)

        real_logits = self.netD(real_images)
        fake_logits = self.netD(fake_images)

        # * max E_{x~P_X}[C(x)] - E_{Z~P_Z}[C(g(z))]
        loss = -(real_logits.mean() - fake_logits.mean())

        loss.backward(retain_graph=True)
        self.optD.step()

        # * Gradient clippling
        for p in self.netD.parameters():
            p.data.clamp_(-self.c, self.c)

        self.metrics["critic-loss"] += [loss.item()]
