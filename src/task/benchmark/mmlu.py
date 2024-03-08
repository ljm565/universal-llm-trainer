import os
from datasets import load_dataset

from utils import LOGGER, DATASET_HELP_MSG, colorstr
from utils.filesys_utils import pickle_save, txt_save
from utils.download_utils import load_data_list


class MassiveMultitaskLanguageUnderstanding:
    def __init__(self, download_path, dataset=None):
        self.dataset = 'lukaemon/mmlu'
        self.names = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', \
                      'clinical_knowledge', 'college_biology', 'college_chemistry', \
                      'college_computer_science', 'college_mathematics', 'college_medicine', \
                      'college_physics', 'computer_security', 'conceptual_physics', 
                      'econometrics', 'electrical_engineering', 'elementary_mathematics', \
                      'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', \
                      'high_school_computer_science', 'high_school_european_history', \
                      'high_school_geography', 'high_school_government_and_politics', \
                      'high_school_macroeconomics', 'high_school_mathematics', \
                      'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', \
                      'high_school_statistics', 'high_school_us_history', 'high_school_world_history', \
                      'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', \
                      'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', \
                      'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', \
                      'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', \
                      'professional_psychology', 'public_relations', 'security_studies', 'sociology', \
                      'us_foreign_policy', 'virology', 'world_religions']
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
            all_datasets = {name: None for name in self.names}
            all_datasets_stats = {name: None for name in self.names}
            max_str_len = 0
            
            for name in self.names:
                datasets = {'train': None, 'validation': None, 'test': None}
                datasets_stats = {}
            
                for k in datasets.keys():
                    try:
                        datasets[k] = load_dataset(self.dataset, name=name, split=k)
                        stat_key, stat_value = f'{k} data length', len(datasets[k])
                        max_str_len = max(max_str_len, len(stat_key))
                    
                    except Exception as e1:
                        try:
                            LOGGER.info(f'Error occured while loading {colorstr(k)} split: {e1} \
                                        \nTrying to load without verifications...')
                            datasets[k] = load_dataset(self.dataset, name=name, split=k, verification_mode='no_checks')
                            stat_key, stat_value = f'{k} data length', len(datasets[k])
                            max_str_len = max(max_str_len, len(stat_key))
                    
                        except Exception as e2:
                            LOGGER.info(f'Error occured while loading {colorstr(k)} split: {e2}')
                            stat_key, stat_value = f'{k} data', None
                            max_str_len = max(max_str_len, len(stat_key))
                    
                    datasets_stats[stat_key] = stat_value
                all_datasets[name] = datasets
                all_datasets_stats[name] = datasets_stats
            
            # check downloaded dataset statistics
            LOGGER.info('\n'+ '#'*20  + ' ' + colorstr(self.dataset+' statistics') + ' ' + '#'*20)
            statistics_msg, formatting = '', '{0:<%ds}' % max_str_len
            for name in self.names:
                for k, v in all_datasets_stats[name].items():
                    if v == None:
                        msg = formatting.format(k)
                        statistics_msg += '{}({}): {}\n'.format(msg, name, v)
                    else:
                        msg = formatting.format(k)
                        statistics_msg += '{}({}): {}\n'.format(msg, name, v)
                statistics_msg += '\n'
            LOGGER.info(statistics_msg.strip())
            LOGGER.info('#'* (53+len(self.dataset)) + '\n')

            # save dataset
            dataset = self.dataset.replace('/', '_')
            data_folder = os.path.join(self.download_path, dataset)
            os.makedirs(data_folder, exist_ok=True)

            pickle_save(os.path.join(data_folder, dataset+'.pkl'), all_datasets)
            txt_save(os.path.join(data_folder, dataset+'_stats.txt'), statistics_msg)
            
            return True

