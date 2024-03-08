import os
import json
import pickle
from pathlib import Path

from utils import LOGGER, colorstr


def find_parent_dir(path: str, n=1):
    for _ in range(n):
        path = os.path.abspath(os.path.dirname(path))
    return path


def pickle_load(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)    


def pickle_save(path: str, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def txt_load(path: str):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def txt_save(path: str, data):
    with open(path, 'w') as f:
        f.write(data)


def json_load(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def json_save(path: str, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def make_project_dir(config, is_rank_zero=False):
    """
    Make project folder.

    Args:
        config: yaml config.
        is_rank_zero (bool): make folder only at the zero-rank device.

    Returns:
        (path): project folder path.
    """
    prefix = colorstr('make project folder')
    project = config.project
    name = config.name

    save_dir = os.path.join(project, name)
    if os.path.exists(save_dir):
        if is_rank_zero:
            LOGGER.info(f'{prefix}: Project {save_dir} already exists. New folder will be created.')
        save_dir = os.path.join(project, name + str(len(os.listdir(project))+1))
    
    if is_rank_zero:
        os.makedirs(project, exist_ok=True)
        os.makedirs(save_dir)
    
    return Path(save_dir)


def yaml_save(file='data.yaml', data=None, header=''):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """

    save_path = Path(file)
    print(data.dumps())
    with open(save_path, "w") as f:
        f.write(data.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")