import sys

from utils.utils import load_config
from utils.log import log

from base.trainer import Trainer
from models.wgan_gp import WGAN_GP
from models.wgan import WGAN
from models.dcgan import DCGAN

from app import app


def start_train_session(Model, cfg_path):
    log.info("Setting up Trainer...")
    trainer = Trainer(Model, cfg_path)
    log.info("Starting Training...")
    trainer.train()


@app.command()
def wgan_gp(cfg_path: str = "./config/config.yml"):
    log.info("WGAN-GP selected for training...")
    start_train_session(WGAN_GP, cfg_path)


@app.command()
def wgan(cfg_path: str = "./config/config.yml"):
    log.info("WGAN selected for training...")
    start_train_session(WGAN, cfg_path)


@app.command()
def dcgan(cfg_path: str = "./config/config.yml"):
    log.info("DCGAN selected for training...")
    start_train_session(DCGAN, cfg_path)


if __name__ == "__main__":
    app()
