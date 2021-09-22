import os
import logging

def make_dir_if_not_exists(
    path : str
):
    """
    verifies whether the given path is a dir and if not creates it
    """
    if not os.path.isdir(path):
        logging.info('creating directory: ./{}'.format(path))
        os.makedirs(path)