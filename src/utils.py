import os
import logging
import json
from typing import List, Tuple

import numpy as np

def make_dir_if_not_exists(path : str) -> bool:
    """
    verifies whether the given path is a dir and if not creates it
    """
    if not os.path.isdir(path):
        logging.info('creating directory: ./{}'.format(path))
        os.makedirs(path)
        return True
    else:
        return False

def split_data(
    data : list,
    val_split : float = 0.15,
    test_split : float = 0.15
) -> Tuple[List, List, List]:
    """Splits the data into 3 splits, train, val and test with the given float splits

    :param data: data on wich to apply the split
    :type data: list
    :param val_split: split of the validation set, defaults to 0.15
    :type val_split: float, optional
    :param test_split: split of the test set, defaults to 0.15
    :type test_split: float, optional
    :return: 3 lists : train, val and test set
    :rtype: Tuple[List, List, List]
    """
    data = np.array(data)
    N = len(data)
    indices = list(range(N))
    np.random.shuffle(indices)

    train_indices = indices[int(N * val_split):-int(N * test_split)]
    val_indices = indices[:int(N * val_split)]
    test_indices = indices[-int(N * test_split):]

    train_set = data[train_indices]
    val_set = data[val_indices]
    test_set = data[test_indices]

    assert len(train_set) + len(val_set) + len(test_set) == N

    return list(train_set), list(val_set), list(test_set)

def update_json(json_file, **kwargs):    
    with open(json_file, 'r') as f:
        data = json.load(f)
        for key, val in kwargs.items():
            if isinstance(val, dict):
                for k,v in val.items():
                    data[key][k] = v
            else:
                data[key] = val

    with open(json_file, 'w') as f:
        json.dump(data, f, indent = 4)

