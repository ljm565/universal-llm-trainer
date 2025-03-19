import os
import re
import random
import platform
import numpy as np
import logging.config
from importlib import metadata
from tqdm import tqdm as tqdm_original

import torch



# Initialize
LOGGING_NAME = f"LLM_TRAINER_{open('version.txt').read().strip()}"
VERBOSE = True
RANK = int(os.getenv('RANK', -1))
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}' if VERBOSE else None  # tqdm bar format


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR

    class ColorFormatter(logging.Formatter):
        """Custom formatter to add colors to log messages using colorstr."""
        def format(self, record):
            if record.levelname == "ERROR":
                record.msg = colorstr("red", record.msg)
            elif record.levelname == "WARNING":
                record.msg = colorstr("yellow", record.msg)
            # elif record.levelname == "INFO":
            #     record.msg = colorstr("green", record.msg)
            elif record.levelname == "DEBUG":
                record.msg = colorstr("blue", record.msg)
            return super().format(record)

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                '()': ColorFormatter,  # Use the custom formatter
                'format': '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
            }
        },
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level
            }
        },
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False
            }
        }
    })
    

set_logging(LOGGING_NAME, verbose=VERBOSE) 
LOGGER = logging.getLogger(LOGGING_NAME)



def colorstr(*input):
    """
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr('blue', 'bold', 'hello world')
        >>> '\033[34m\033[1mhello world\033[0m'
    """
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']



def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            LOGGER.warning('WARNING ⚠️ Upgrade to torch>=2.0.0 for deterministic training.')
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False



def parse_version(version='0.0.0') -> tuple:
    """
    Convert a version string to a tuple of integers, ignoring any extra non-numeric string attached to the version. This
    function replaces deprecated 'pkg_resources.parse_version(v)'.

    Args:
        version (str): Version string, i.e. '2.0.1+cpu'

    Returns:
        (tuple): Tuple of integers representing the numeric part of the version and the extra string, i.e. (2, 0, 1)
    """
    try:
        return tuple(map(int, re.findall(r'\d+', version)[:3]))  # '2.0.1+cpu' -> (2, 0, 1)
    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}')
        return 0, 0, 0


def emojis(string=''):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode('ascii', 'ignore') if WINDOWS else string


def check_version(current: str = '0.0.0',
                  required: str = '0.0.0',
                  name: str = 'version',
                  hard: bool = False,
                  verbose: bool = False) -> bool:
    """
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        name (str, optional): Name to be used in warning message.
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.
        verbose (bool, optional): If True, print warning message if requirement is not met.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current='22.04', required='==22.04')

        # Check if current version is greater than or equal to 22.04
        check_version(current='22.10', required='22.04')  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current='22.04', required='<=22.04')

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current='21.10', required='>20.04,<22.04')
        ```
    """
    if not current:  # if current is '' or None
        LOGGER.warning(f'WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.')
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            name = current  # assigned package name to 'name' arg
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError:
            if hard:
                raise ModuleNotFoundError(emojis(f'WARNING ⚠️ {current} package is required but not installed'))
            else:
                return False

    if not required:  # if required is '' or None
        return True

    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)
    for r in required.strip(',').split(','):
        op, v = re.match(r'([^0-9]*)([\d.]+)', r).groups()  # split '>=22.04' -> ('>=', '22.04')
        v = parse_version(v)  # '1.2.3' -> (1, 2, 3)
        if op == '==' and c != v:
            result = False
        elif op == '!=' and c == v:
            result = False
        elif op in ('>=', '') and not (c >= v):  # if no constraint passed assume '>=required'
            result = False
        elif op == '<=' and not (c <= v):
            result = False
        elif op == '>' and not (c > v):
            result = False
        elif op == '<' and not (c < v):
            result = False
    if not result:
        warning_message = f'WARNING ⚠️ {name}{op}{required} is required, but {name}=={current} is currently installed'
        if hard:
            raise ModuleNotFoundError(emojis(warning_message))  # assert version requirements met
        if verbose:
            LOGGER.warning(warning_message)
    return result



TORCH_2_0 = check_version(torch.__version__, '2.0.0')



class TQDM(tqdm_original):
    """
    Custom Ultralytics tqdm class with different default arguments.

    Args:
        *args (list): Positional arguments passed to original tqdm.
        **kwargs (dict): Keyword arguments, with custom defaults applied.
    """

    def __init__(self, *args, **kwargs):
        """Initialize custom Ultralytics tqdm class with different default arguments."""
        # Set new default values (these can still be overridden when calling TQDM)
        kwargs['disable'] = not VERBOSE or kwargs.get('disable', False)  # logical 'and' with default value if passed
        kwargs.setdefault('bar_format', TQDM_BAR_FORMAT)  # override default value if passed
        super().__init__(*args, **kwargs)


def log_if_rank_zero(func):
    def wrapper(self, *args, **kwargs):
        if self.is_rank_zero:
            return func(self, *args, **kwargs)
    return wrapper


@log_if_rank_zero
def print_mem_consumption(self, path):
    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
    LOGGER.info(f'{colorstr(path)} accounts for {colorstr(mem)} of GPU memory.')


@log_if_rank_zero
def logger(self, message):
    LOGGER.info(colorstr(message))


########################## MSG ##########################
DATASET_HELP_MSG = 'If you are looking for LLM benchmark datasets, you can choose from the following list: [gsm8k, ai2_arc, Rowan/hellaswag, winogrande, lukaemon/mmlu, truthful_qa]'
OPTIM_CRITERION_MSG = 'optimizer_step_criterion must be belonged to' + f' [{colorstr("epoch")}, {colorstr("step")}] '
SCHEDULER_MSG = 'scheduler_type must be belonged to' + f' [{colorstr("linear")}, {colorstr("cosine")}] '
FSDP_WRAP_MSG = 'wrap_policy must be belonged to' + f' [{colorstr("size_based")}, {colorstr("transformer_based")}] '
DATASET_TRAIN_TYPE_MSG = 'Currently, we only support' + f' [{colorstr("qa")}, {colorstr("ar")}] ' + 'as dataset_train_type'

##################### Assertion List #####################
OPTIM_CRITERION = ['epoch', 'step']
SCHEDULER_TYPE = ['linear', 'cosine']
FSDP_WRAP_TYPE = ['size_based', 'transformer_based']