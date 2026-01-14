import os
import yaml
import json
import pickle
from pathlib import Path
from typing import Any, List
from dataclasses import asdict, is_dataclass
from ruamel.yaml.comments import CommentedSeq, CommentedMap
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarstring import ScalarString

from univlt.config import TrainingConfig
from univlt.utils import is_rank_zero, colorstr, log



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



def make_project_dir(cfg: TrainingConfig) -> Path:
    """
    Make project folder.

    Args:
        cfg (TrainingConfig): Training configurations.

    Returns:
        (path): project folder path.
    """
    prefix = colorstr('make project folder')
    project = cfg.project
    name = cfg.name

    save_dir = os.path.join(project, name)
    if os.path.exists(save_dir):
        log(f'{prefix}: Project {save_dir} already exists. New folder will be created.')
        save_dir = os.path.join(project, name + str(len(os.listdir(project))+1))
    
    if is_rank_zero['value']:
        os.makedirs(project, exist_ok=True)
        os.makedirs(save_dir)
    
    return Path(save_dir)



def to_builtin(x: Any) -> Any:
    """
    Recursively convert non-builtin Python objects into YAML/JSON-serializable builtin types.

    Args:
        x (Any): An arbitrary Python object to be converted into builtin types.

    Returns:
        Any: A YAML/JSON-serializable object composed only of builtin Python types (dict, list, str, int, float, bool, None).
    """
    # dataclass → dict
    if is_dataclass(x):
        x = asdict(x)

    # ruamel sequence / mapping
    if isinstance(x, CommentedSeq):
        return [to_builtin(v) for v in list(x)]

    if isinstance(x, CommentedMap):
        return {str(k): to_builtin(v) for k, v in dict(x).items()}

    # ruamel scalar types
    if isinstance(x, ScalarFloat):
        return float(x)

    if isinstance(x, ScalarInt):
        return int(x)

    if isinstance(x, ScalarString):
        return str(x)

    # dict
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}

    # list / tuple / set
    if isinstance(x, (list, tuple, set)):
        return [to_builtin(v) for v in x]

    # Path → str
    if isinstance(x, Path):
        return str(x)

    # class / type → name
    if isinstance(x, type):
        return x.__name__

    return x



def yaml_save(
        file: str = 'data.yaml', 
        data: Any = None
    ):
    """
    Save data to an YAML file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (Any, optional): Data to save in YAML format.
    """
    if data is None:
        raise ValueError(colorstr("red", "data must be provided"))
    
    payload = to_builtin(data)
    save_path = Path(file)
    log(payload)

    with open(save_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    log(f"Config is saved at {save_path}")
