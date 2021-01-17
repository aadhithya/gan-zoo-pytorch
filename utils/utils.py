import yaml
from munch import DefaultMunch
from utils.log import log


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info("Loading Config")
    return DefaultMunch.fromDict(cfg)
