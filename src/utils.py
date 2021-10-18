import os
import logging

import numpy as np

def make_dir_if_not_exists(
    path : str
):
    """
    verifies whether the given path is a dir and if not creates it
    """
    if not os.path.isdir(path):
        logging.info('creating directory: ./{}'.format(path))
        os.makedirs(path)

def split_data(
    data : list,
    val_split : float = 0.15,
    test_split : float = 0.15
):
    """
    Splits the input data into train, validation and test datasets
    """
    data = np.array(data)
    N = len(data)
    indices = list(range(N))
    np.random.shuffle(indices)

    train_indices = indices[int(N * val_split):int(N * test_split)]
    val_indices = indices[:int(N * val_split)]
    test_indices = indices[int(N * test_split):]

    train_set = data[train_indices]
    val_set = data[val_indices]
    test_set = data[test_indices]

    return list(train_set), list(val_set), list(test_set)