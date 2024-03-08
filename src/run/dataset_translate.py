import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from task import TranslationTask
from utils import LOGGER, DATASET_HELP_MSG, colorstr



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_folder_path', type=str, required=True, help=DATASET_HELP_MSG)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--src', default='en', type=str)
    parser.add_argument('--trg', default='ko', type=str)
    parser.add_argument('--nmt', default='google', type=str)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    dataset = args.dataset_folder_path[args.dataset_folder_path.rfind('/')+1:]
    dataset_folder_path = os.path.abspath(args.dataset_folder_path)
    save_path = os.path.abspath(args.save_path)
    status = TranslationTask(dataset_folder_path, save_path, args.src, args.trg, args.nmt).translate(args.verbose)
    if status:
        dataset = colorstr(dataset)
        LOGGER.info(f'{dataset} is translated successfully.')
    else:
        dataset = colorstr('red', dataset)
        LOGGER.info(f'{dataset} translation is failed.')