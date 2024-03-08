import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from task import DownloadTask
from utils import LOGGER, DATASET_HELP_MSG, colorstr



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help=DATASET_HELP_MSG)
    parser.add_argument('--download_path', type=str, required=True)
    args = parser.parse_args()

    download_path = os.path.abspath(args.download_path) 
    status = DownloadTask(args.dataset, download_path).download()
    if status:
        dataset = colorstr(args.dataset)
        LOGGER.info(f'{dataset} is downloaded successfully.')
    else:
        dataset = colorstr('red', args.dataset)
        LOGGER.info(f'{dataset} donwloading is failed.')
