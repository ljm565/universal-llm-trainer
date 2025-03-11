import random
import numpy as np

import torch

from data_collection import (
    AutoregressiveDataset,
    ARCDataset,
    QADataset,
)
from utils import LOGGER, DATASET_TRAIN_TYPE_MSG


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def choose_proper_dataset(dataset_name):
    if 'qa' == dataset_name.lower():
        return QADataset
    elif 'ar' == dataset_name.lower():
        return AutoregressiveDataset
    else:
        raise ValueError(LOGGER.error(f'Invalid dataset name: {dataset_name}\n{DATASET_TRAIN_TYPE_MSG}'))