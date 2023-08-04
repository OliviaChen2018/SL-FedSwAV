import logging
import torch
import sys

def setup_logger(name, log_file, level=logging.INFO, console_out = True):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file, mode='a')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(handler)
    if console_out:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
    return logger

def distributed_concat(tensor, num_total_examples, world_size):
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]