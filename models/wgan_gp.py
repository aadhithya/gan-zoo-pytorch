import torch
import torch.nn as nn
from torch.optim import Adam
from torch import autograd

from base.model import BaseGAN
from models.modules.net import NetG, NetD

class WGAN_GP(BaseGAN):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.netG = NetG(self.cfg.z_dim, self.cfg.img_ch)
        self.netD = NetD(self.cfg.img_ch)

        self.n_critic = 5

        self._update_model_optimizers()

    def _update_model_optimizers(self):
        self.netG = self.netG.to(self.cfg.device)
        self.netD = self.netD.to(self.cfg.device)

        self.optG = Adam(self.netG.parameters(), lr=self.cfg.lr.g)
        self.optD = Adam(self.netD.parameters(), lr=self.cfg.lr.d)

    def generator_step(self, data):
        self.netG.train()
        self.netD.eval

        self.optG.zero_grad()

        noise = self.sample_noise()

        fake_images = self.netG(noise)

        fake_logits = self.netD(fake_images)

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

        gradient_penalty = self.cfg.w_gp * self._compute_gp(real_images, fake_images)
        loss_c = fake_logits.mean() - real_logits.mean()

        loss = loss_c + gradient_penalty

        loss.backward()
        self.optD.step()

        self.metrics["critic-loss"] += [loss.item()]
        self.metrics["gp"] += [gradient_penalty.item()]
    
    def _compute_gp(self, real_data, fake_data):
        batch_size = real_data.size(0)
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device())
        eps = eps.expand_as(real_data)

        interpolation = eps * real_data + (1 - eps) * fake_data

        interp_logits = self.netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

