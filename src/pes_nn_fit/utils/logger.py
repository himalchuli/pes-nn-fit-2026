import logging
import os
from datetime import datetime
import sys
import torch
from logging.handlers import RotatingFileHandler


def setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    log_path = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # ---- FILE HANDLER (DEBUG level) ----
    fh = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5  # 5 MB  # keep last 5 logs
    )
    fh.setLevel(logging.DEBUG)

    # ---- CONSOLE HANDLER (INFO level) ----
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # ---- FORMATTER ----
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logger.info(f"Logger initialized. Saving logs to {log_path}")

    # Environment info
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    logger.debug("Environment: %s", env_name)
    logger.debug("Python version: %s", sys.version.split()[0])
    logger.debug("PyTorch version: %s", torch.__version__)
    logger.debug("Device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    return logger
