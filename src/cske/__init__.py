import os
import logging
import sys

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from .extractor import CSKE
from .model import KeyBERTMod
from .filter import KeywordFilter

logger = logging.getLogger("cske")
logger.addHandler(logging.NullHandler())

def enable_logging(level: int = logging.INFO):
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers if not isinstance(h, logging.NullHandler)):
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)

enable_logging()

__version__ = "0.3.0"
__all__ = ["CSKE", "KeyBERTMod", "KeywordFilter", "enable_logging"]