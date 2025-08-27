# logger_config.py

import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Optional: configure logger only if not already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger