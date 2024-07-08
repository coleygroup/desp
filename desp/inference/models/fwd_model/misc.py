"""
Taken from ASKCOSv2 template relevance module:
https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance/-/blob/main/misc.py?ref_type=heads
"""

import logging
import os
import sys
import torch


class BlockPrint:
    """Context manager to suppress printing from subprocesses"""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def log_args(args, message: str):
    log_rank_0(message)
    for k, v in vars(args).items():
        log_rank_0(f"**** {k} = *{v}*")


def log_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logging.info(message)
            sys.stdout.flush()
    else:
        logging.info(message)
        sys.stdout.flush()


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler(sys.stdout)
    fh.setLevel(logging.INFO)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
