import os
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from .extractor import KeyExGen as CSKE
from .model import KeyBERTMod
from .filter import KeywordFilter
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

def enable_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger("cske")
    logger.addHandler(handler)
    logger.setLevel(level)

__version__ = "0.1.1"
__all__ = ["CSKE", "KeyBERTMod", "KeywordFilter"]