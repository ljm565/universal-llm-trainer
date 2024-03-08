import os
from datasets import list_datasets

from .filesys_utils import pickle_load, pickle_save


def load_data_list(folder_to_check_cache=None):
    """
    Args:
        folder_to_check_cache (str): Path to the folder where the cache file is located.
                                    If cache does not exist, it will be created.

    Returns:
        data_list (List[str]): List of dataset names.
    """
    if folder_to_check_cache:
        os.makedirs(folder_to_check_cache, exist_ok=True)
        data_list_cache_file_path = os.path.join(folder_to_check_cache, '.data_list.pkl')
        
        if os.path.exists(data_list_cache_file_path):
            data_list = pickle_load(data_list_cache_file_path)
        else:
            data_list = list_datasets()
            pickle_save(data_list_cache_file_path, data_list)
    
    else:
        data_list = list_datasets()
    
    return data_list