import os
import json
import pickle
from pathlib import Path
from typing import Any, List

from utils import is_rank_zero, colorstr, log



def pickle_load(path: str) -> Any:
    """
    Load pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Any: The object loaded from the pickle file.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)    



def pickle_save(path: str, data: Any) -> None:
    """
    Save data to a pickle file.

    Args:
        path (str): Path to the pickle file.
        data (Any): Data to save.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)



def txt_load(path: str) -> List[str]:
    """
    Load data from a text file.
    For easy processing(e.g. ARC templates, etc.), it returns a list of lines without newline characters.

    Args:
        path (str): Path to the text file.

    Returns:
        List[str]: List of lines in the text file.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines



def txt_save(path: str, data: str) -> None:
    """
    Save data to a text file.

    Args:
        path (str): Path to the text file.
        data (str): Data to save.
    """
    with open(path, 'w') as f:
        f.write(data)



def json_load(path: str) -> dict:
    """
    Load json file.

    Args:
        path (str): Path to the json file.

    Returns:
        dict: The object loaded from the json file.
    """
    with open(path, 'r') as f:
        return json.load(f)



def json_save(path: str, data: dict) -> None:
    """
    Save json file.

    Args:
        path (str): Path to the json file.
        data (dict): Data to save.
    """
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



def make_project_dir(config) -> Path:
    """
    Make project folder.

    Args:
        config: yaml config.

    Returns:
        (path): project folder path.
    """
    prefix = colorstr('make project folder')
    project = config.project
    name = config.name

    save_dir = os.path.join(project, name)
    if os.path.exists(save_dir):
        log(f'{prefix}: Project {save_dir} already exists. New folder will be created.')
        save_dir = os.path.join(project, name + str(len(os.listdir(project))+1))
    
    if is_rank_zero['value']:
        os.makedirs(project, exist_ok=True)
        os.makedirs(save_dir)
    
    return Path(save_dir)



def yaml_save(file:str='data.yaml', data:Any=None) -> None:
    """
    Save data to an YAML file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (Any, optional): Data to save in YAML format.
    """
    save_path = Path(file)
    log(data.dumps())
    with open(save_path, "w") as f:
        f.write(data.dumps(modified_color=None, quote_str=True))
        log(f"Config is saved at {save_path}")
