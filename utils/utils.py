import yaml
from munch import DefaultMunch
from utils.log import log

import torch.nn as nn


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Loading Config")
    return DefaultMunch.fromDict(cfg)


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        # nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_uniform_(m.weight)
        nn.init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            if m.bias.data is not None:
                m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
