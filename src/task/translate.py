import os
from tqdm import tqdm
from copy import deepcopy

from utils import LOGGER, colorstr
from utils.filesys_utils import pickle_save
from tools import LangTranslator
from task.benchmark import (
    BaseTask,
    ARC,
    GradeSchoolMath8k,
    TruthfulQA,
    HellaSwag,
    WinoGrande,
    MassiveMultitaskLanguageUnderstanding,
)



class TranslationTask:
    def __init__(self, dataset_folder_path, save_path, src, trg, nmt='google'):
        self.dataset_folder_path = dataset_folder_path
        self.save_path = save_path
        self.src, self.trg = src, trg
        self.nmt = nmt
        self.dataset = dataset_folder_path[dataset_folder_path.rfind('/')+1:]
        self.translator = LangTranslator(self.src, self.trg, self.nmt)


    def prepare_dataset(self):
        LOGGER.info(f'Preparing {colorstr(self.dataset)} datasets for translation...')
        
        if 'ai2_arc' in self.dataset_folder_path:
            dataset_manager = ARC(self.save_path, dataset=self.dataset, forTranslation=True)
            return dataset_manager.prepare_translation(self.dataset_folder_path)
        elif self.dataset == 'gsm8k':
            dataset_manager = GradeSchoolMath8k(self.save_path, dataset=self.dataset, forTranslation=True)
            return dataset_manager.prepare_translation(self.dataset_folder_path)
        elif self.dataset == 'truthful_qa':
            dataset_manager = TruthfulQA(self.save_path, dataset=self.dataset, forTranslation=True)
            return dataset_manager.prepare_translation(self.dataset_folder_path)
        elif self.dataset == 'Rowan/hellaswag':
            return HellaSwag(self.download_path, self.dataset).download_dataset()
        elif self.dataset == 'winogrande':
            return WinoGrande(self.download_path, self.dataset).download_dataset()
        elif self.dataset == 'lukaemon/mmlu':
            return MassiveMultitaskLanguageUnderstanding(self.download_path, self.dataset).download_dataset()
        else:
            return BaseTask(self.dataset, self.download_path).download_dataset()


    def translate(self, verbose=False):
        # def _change_nested_value(d, keys, verbose):
        #     """
        #     Change the value in a nested dictionary at the specified depth.

        #     Args:
        #         d: The input dictionary.
        #         keys: A list of keys representing the path to the desired depth.

        #     Example:
        #     change_nested_value(my_dict, ['key1', 'key2', 'key3'])
        #     """
        #     if len(keys) == 1:
        #         instance = d[keys[0]]
        #         if isinstance(instance, list):
        #             assert all([isinstance(i, str) for i in instance]), f'Values in the list must be str type, but one or more values are not str type. Values: {instance}'
        #             d[keys[0]] = [self.translator.translate(i) for i in instance]
        #         elif isinstance(instance, str):
        #             assert isinstance(instance, str), f'Value must be str type, but got {type(instance)} type.'
        #             d[keys[0]] = self.translator.translate(instance)
                
        #         if verbose and self.idx % 50 == 0:
        #             if isinstance(instance, list):
        #                 for s, t in zip(instance, d[keys[0]]):
        #                     print(f'{s} -> {t}')
        #             else:
        #                 print(f'{instance} -> {d[keys[0]]}')

        #     else:
        #         key = keys[0]
        #         if key in d:
        #             _change_nested_value(d[key], keys[1:], verbose)
        #         else:
        #             raise KeyError(f"Key not found: {key}")

        def _change_nested_value(d, keys, verbose):
            """
            Change the value in a nested dictionary at the specified depth.

            Args:
                d: The input dictionary.
                keys: A list of keys representing the path to the desired depth.

            Example:
            change_nested_value(my_dict, ['key1', 'key2', 'key3'])
            """
            for i, key in enumerate(keys):
                if key not in d:
                    continue

                if i == len(keys)-1:
                    if isinstance(d[key], list):
                        assert all([isinstance(i, str) for i in d[key]]), f'Values in the list must be str type, but one or more values are not str type. Values: {d}'
                        source = deepcopy(d[key])
                        d[key] = [self.translator.translate(i)[1] for i in d[key]]
                    elif isinstance(d[key], str):
                        assert isinstance(d[key], str), f'Value must be str type, but got {type(d[key])} type.'
                        source, result = self.translator.translate(d[key])
                        d[key] = result
                    
                    if verbose and self.idx % 50 == 0:
                        if isinstance(d, list):
                            for s, t in zip(source, d[key]):
                                print(f'{s} -> {t}')
                        else:
                            print(f'{source} -> {d[key]}')
                else:
                    d = d[key]
            
                
        dataset, flags = self.prepare_dataset()
        LOGGER.info(f'{colorstr(self.dataset)} datasets translation started({colorstr(self.src)} -> {colorstr(self.trg)})...')
        
        # translation
        for name in dataset.keys():
            for state in ['train', 'validation', 'test']:
                self.idx = 0
                single_dataset = dataset[name][state]
                
                # None case
                if not single_dataset:
                    continue
                
                single_dataset = list(single_dataset)
                for data in tqdm(single_dataset, desc=f'{name}.{state}...'):
                    for flag in flags:
                        _change_nested_value(data, flag.split('.'), verbose)
                    self.idx += 1
                
                dataset[name][state] = single_dataset

        # save
        file_name = f'{self.dataset}_{self.trg}'
        os.makedirs(os.path.join(self.save_path, file_name), exist_ok=True)
        save_path = os.path.join(*[self.save_path, f'{file_name}', f'{file_name}.pkl'])
        pickle_save(save_path, dataset)