import sys

from utils.utils import load_config
from utils.log import log

from base.trainer import Trainer
from models.wgan_gp import WGAN_GP

# import typer
from app import app


@app.command()
def wgan_gp(cfg_path: str = "./config/config.yml"):
    log.info("Setting up Trainer...")
    trainer = Trainer(WGAN_GP, cfg_path)
    log.info("Starting Training...")
    trainer.train()


@app.command()
def model2(cfg_path: str):
    pass


if __name__ == "__main__":
    app()
