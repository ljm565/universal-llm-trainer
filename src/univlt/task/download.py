from task.benchmark import (
    BaseTask,
    ARC,
    GradeSchoolMath8k,
    TruthfulQA,
    HellaSwag,
    WinoGrande,
    MassiveMultitaskLanguageUnderstanding,
)



class DownloadTask:
    def __init__(self, dataset, download_path):
        self.dataset = dataset
        self.download_path = download_path


    def download(self):
        if self.dataset == 'allenai/ai2_arc':
            return ARC(self.download_path, self.dataset).download_dataset()
        elif self.dataset == 'gsm8k':
            return GradeSchoolMath8k(self.download_path, self.dataset).download_dataset()
        elif self.dataset == 'truthful_qa':
            return TruthfulQA(self.download_path, self.dataset).download_dataset()
        elif self.dataset == 'Rowan/hellaswag':
            return HellaSwag(self.download_path, self.dataset).download_dataset()
        elif self.dataset == 'winogrande':
            return WinoGrande(self.download_path, self.dataset).download_dataset()
        elif self.dataset == 'lukaemon/mmlu':
            return MassiveMultitaskLanguageUnderstanding(self.download_path, self.dataset).download_dataset()
        else:
            return BaseTask(self.download_path, self.dataset).download_dataset()

