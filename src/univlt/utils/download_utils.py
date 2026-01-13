import os
from typing import List
from datasets import list_datasets

from .filesys_utils import pickle_load, pickle_save



def load_data_list(folder_to_check_cache:str=None) -> List[str]:
    """
    To check download availability, load the list of datasets from the cache file.

    Args:
        folder_to_check_cache (str, optional): Path to the folder where the cache file is located. 
                                               If cache does not exist, the cache will be created. Defaults to None.

    Returns:
        List[str]: List of dataset names
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
