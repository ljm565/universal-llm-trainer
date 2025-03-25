import random
import numpy as np
from typing import Type, Union

import torch

from data_collection import (
    AutoregressiveDataset,
    ARCDataset,
    QADataset,
)
from utils import log, DATASET_TRAIN_TYPE_MSG



def seed_worker(worker_id):  # noqa
    """
    Set the random seed for the DataLoader worker process.

    This function ensures reproducibility by setting the seed for the random 
    number generators used in PyTorch, NumPy, and Python's `random` module 
    for each worker process. The seed is derived from PyTorch's initial seed.

    Args:
        worker_id (int): The ID of the DataLoader worker process.

    Notes:
        - This function is commonly used as the `worker_init_fn` argument 
          in PyTorch's DataLoader.
        - For more information, see the PyTorch documentation:
          https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def choose_proper_dataset(dataset_name: str) -> Type[Union[QADataset, AutoregressiveDataset, ARCDataset]]:
    """
    Select the appropriate dataset class based on the given dataset name.

    Args:
        dataset_name (str): Name of the dataset. Expected values are 'qa', 'ar', or 'arc' (case-insensitive).

    Returns:
        Type[Union[QADataset, AutoregressiveDataset, ARCDataset]]: 
            The class corresponding to the specified dataset name.

    Raises:
        ValueError: If the provided dataset name is invalid.

    Example:
        >>> dataset_class = choose_proper_dataset("qa")
        >>> print(dataset_class)
        <class 'data_collection.qa_dataset.QADataset'>
    """
    if 'qa' == dataset_name.lower():
        return QADataset
    elif 'ar' == dataset_name.lower():
        return AutoregressiveDataset
    elif 'arc' == dataset_name.lower():
        return ARCDataset
    else:
        raise ValueError(log(f'Invalid dataset name: {dataset_name}\n{DATASET_TRAIN_TYPE_MSG}', level='error'))
