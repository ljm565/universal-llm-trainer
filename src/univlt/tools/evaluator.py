import numpy as np
from rouge import Rouge
from nltk import edit_distance
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score



class Evaluator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


    def cal_bleu_score(self, pred, gt, n=4):
        assert (isinstance(pred, str) and isinstance(gt, str)) or (isinstance(pred, list) and isinstance(gt, list))    
        if isinstance(pred, str):
            pred, gt = [pred], [gt]
        weights = tuple([1/n] * n)
        pred = [self.tokenizer.tokenize(text) for text in pred]
        gt = [[self.tokenizer.tokenize(text)] for text in gt]
        if isinstance(pred, str):
            return sentence_bleu(gt[0], pred[0], weights=weights)
        return corpus_bleu(gt, pred, weights=weights)
    

    def cal_rouge_score(self, pred, gt, n=4):
        assert (isinstance(pred, str) and isinstance(gt, str)) or (isinstance(pred, list) and isinstance(gt, list))    
        if isinstance(pred, str):
            pred, gt = [pred], [gt]
        
        if not hasattr(self, 'rouge'):
            if n == None:
                self.rouge = Rouge(['rouge-l'])
            else:
                self.rouge = Rouge([f'rouge-{n}'])
        
        rouge = self.rouge.get_scores(pred, gt, avg=True)
        if 'rouge-l' in rouge:
            return rouge['rouge-l']['f']
        return rouge[f'rouge-{n}']['f']
    

    def cal_meteor_score(self, pred, gt):
        assert (isinstance(pred, str) and isinstance(gt, str)) or (isinstance(pred, list) and isinstance(gt, list))
        if isinstance(pred, str):
            pred, gt = [pred], [gt]
        pred = [self.tokenizer.tokenize(text) for text in pred]
        gt = [[self.tokenizer.tokenize(text)] for text in gt]
        meteor_scores = [meteor_score(g, p) for g, p in zip(gt, pred)]
        return sum(meteor_scores) / len(meteor_scores)
    

    def cal_edit_distance(self, pred, gt):
        assert (isinstance(pred, str) and isinstance(gt, str)) or (isinstance(pred, list) and isinstance(gt, list))
        if isinstance(pred, str):
            return edit_distance(pred, gt) / max(len(pred), len(gt))
        return sum([edit_distance(p, g) / max(len(p), len(g)) for p, g in zip(pred, gt)]) / len(pred)
    

    def cal_ppl(self, loss):
        return np.exp(loss)
    

    