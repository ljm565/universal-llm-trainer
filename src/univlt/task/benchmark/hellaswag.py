import os
from datasets import load_dataset

from utils import LOGGER, DATASET_HELP_MSG, colorstr
from utils.filesys_utils import  pickle_save, txt_save
from utils.download_utils import load_data_list


class HellaSwag:
    def __init__(self, download_path, dataset=None):
        self.dataset = 'Rowan/hellaswag'
        self.names = None
        if dataset:
            assert self.dataset == dataset
        self.download_path = download_path
        self.data_list = load_data_list(self.download_path)
    

    def download_dataset(self):
        """
        Returns:
            (bool): True if the dataset is downloaded successfully.
        """
        if self.dataset not in self.data_list:
            dataset = colorstr('red', self.dataset)
            LOGGER.info(f'{dataset} is not available in these version of HuggingFace datasets. Please check the version or dataset name.')
            LOGGER.info(DATASET_HELP_MSG)
            return False
        
        else:
            LOGGER.info(f'Downloading {colorstr(self.dataset)}...')
            # download dataset
            max_str_len = 0
            datasets = {'train': None, 'validation': None, 'test': None}
            datasets_stats = {}

            for k in datasets.keys():
                try:
                    datasets[k] = load_dataset(self.dataset, split=k)
                    stat_key, stat_value = f'{k} data length', len(datasets[k])
                    max_str_len = max(max_str_len, len(stat_key))
                
                except Exception as e1:
                    try:
                        LOGGER.info(f'Error occured while loading {colorstr(k)} split: {e1} \
                                    \nTrying to load without verifications...')
                        datasets[k] = load_dataset(self.dataset, split=k, verification_mode='no_checks')
                        stat_key, stat_value = f'{k} data length', len(datasets[k])
                        max_str_len = max(max_str_len, len(stat_key))
                
                    except Exception as e2:
                        LOGGER.info(f'Error occured while loading {colorstr(k)} split: {e2}')
                        stat_key, stat_value = f'{k} data', None
                        max_str_len = max(max_str_len, len(stat_key))
                
                datasets_stats[stat_key] = stat_value
            
            # check downloaded dataset statistics
            LOGGER.info('\n'+ '#'*20  + ' ' + colorstr(self.dataset+' statistics') + ' ' + '#'*20)
            statistics_msg, formatting = '', '{0:<%ds}' % max_str_len
            for k, v in datasets_stats.items():
                if v == None:
                    msg = formatting.format(k)
                    statistics_msg += '{}: {}\n'.format(msg, v)
                else:
                    msg = formatting.format(k)
                    statistics_msg += '{}: {}\n'.format(msg, v)
            LOGGER.info(statistics_msg.strip())
            LOGGER.info('#'* (53+len(self.dataset)) + '\n')

            # save dataset
            dataset = self.dataset.replace('/', '_')
            data_folder = os.path.join(self.download_path, dataset)
            os.makedirs(data_folder, exist_ok=True)
            
            pickle_save(os.path.join(data_folder, dataset+'.pkl'), {'hellaswag': datasets})
            txt_save(os.path.join(data_folder, dataset+'_stats.txt'), statistics_msg)
            
            return True

