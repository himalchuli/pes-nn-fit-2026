import os
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.debug("Global seed set to %s", seed)
