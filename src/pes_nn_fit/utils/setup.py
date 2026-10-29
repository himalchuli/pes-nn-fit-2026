from pathlib import Path

import torch
import yaml

from pes_nn_fit.utils.helpers import merge_config
from pes_nn_fit.utils.logger import setup_logger
from pes_nn_fit.utils.seed import set_seed

DEFAULT_CONFIG_PATH = "./src/pes_nn_fit/config/default.yaml"


def setup_experiment(config_path: str):
    """Load config merge with default, set seed, initialise logger and device."""
    config_path = Path(config_path).resolve()
    exp_root = config_path.parent

    # load the config file and default config
    with open(config_path) as f:
        user_config = yaml.safe_load(f)
    with open(DEFAULT_CONFIG_PATH) as f:
        default_config = yaml.safe_load(f)

    # merge the config with the default config
    config = merge_config(default_config, user_config)

    set_seed(config["runtime"]["seed"])

    logger = setup_logger(exp_root / "logs")
    logger.debug(
        "Configuration:\n%s",
        yaml.dump(config, sort_keys=False, default_flow_style=False),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return config, logger, device, exp_root
