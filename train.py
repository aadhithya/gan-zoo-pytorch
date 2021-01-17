import argparse
import sys

from utils.utils import load_config
from utils.log import log

from base.trainer import Trainer
from models.wgan_gp import WGAN_GP

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfg_path = sys.argv[1]
    else:
        cfg_path = "./config/config.yml"
