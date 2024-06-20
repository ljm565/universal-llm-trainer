import random
import numpy as np

import torch

from data_collection import (
    ARCDataset,
    QADataset,
)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def choose_proper_dataset(dataset_name):
    if 'arc' in dataset_name.lower():
        return ARCDataset
    elif dataset_name.lower() in ['kopolyglot_easy', 'kopolyglot_easy_v2']:
        return QADataset
    else:
        raise NotImplementedError