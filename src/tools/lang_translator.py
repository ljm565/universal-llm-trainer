import time
from googletrans import Translator
from easynmt import EasyNMT

from utils import LOGGER, colorstr



class LangTranslator:
    def __init__(self, src='en', trg='ko', nmt='google'):
        assert nmt in ['google', 'easynmt']
        self.src, self.trg = src, trg
        self.nmt = nmt
        self.translator = Translator() if self.nmt == 'google' else EasyNMT('m2m_100_1.2B')

    
    def translate(self, text: str):
        if not isinstance(text, str):
            LOGGER.info(f'{colorstr("red", "Input must be a string")}: {text}')
            return '', '' 

        if self.nmt == 'google':
            return self._google_translate(text)
        elif self.nmt == 'easynmt':
            return self._easynmt_translate(text)
        
    
    def _google_translate(self, text: str):
        try:
            output = self.translator.translate(text, src=self.src, dest=self.trg)
            return output.origin, self.postprocess(output.text, self.trg)
        except:
            try:
                time.sleep(1)
                self.translator = Translator()
                output = self.translator.translate(text, src=self.src, dest=self.trg)
                return output.origin, self.postprocess(output.text, self.trg)
            except:
                LOGGER.info(f'{colorstr("red", "Translation failed, Try to use EasyNMT")}: {text}')
                if not hasattr(self, 'temporal_translator'):
                    self.temporal_translator = EasyNMT('m2m_100_1.2B')
                original, output = self._easynmt_translate(text, self.temporal_translator)
                return original, output
    

    def _easynmt_translate(self, text: str, temporal_translator=None):
        if temporal_translator == None:
            try:
                return text, self.translator.translate(text, target_lang=self.trg)
            except:
                LOGGER.info(f'{colorstr("red", "Translation failed")}: {text}')
                return text, text
        else:
            try:
                return text, temporal_translator.translate(text, target_lang=self.trg)
            except:
                LOGGER.info(f'{colorstr("red", "Translation failed")}: {text}')
                return text, text

    
    def postprocess(self, text, trg):
        if trg == 'ko':
            punctuation = ['.', ',', '?', '!']
            for p in punctuation:
                text = text.replace(p, p + ' ')
                text = ' '.join(text.split())
        return text



# if __name__ == '__main__':
#     translator = LangTranslator()

#     text = 'George "wants" to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? dry palm'

#     result = translator.translate(text)
#     print(result)
