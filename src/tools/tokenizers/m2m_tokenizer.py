from transformers import AutoTokenizer



class M2M100Tokenizer:
    def __init__(self, path, src_code, trg_code):
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # special tokens
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.bos_token, self.bos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.eos_token, self.tokenizer.eos_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id
        self.src_token, self.src_token_id = f'__{src_code}__', self.tokenizer.convert_tokens_to_ids(f'__{src_code}__')
        self.trg_token, self.trg_token_id = f'__{trg_code}__', self.tokenizer.convert_tokens_to_ids(f'__{trg_code}__') 

        # vocab size
        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        """
        Args:
            s (str): text.

        Returns:
            (list): tokenized text.
        """
        return self.tokenizer.tokenize(s)


    def encode(self, s, lang='en'):
        """
        Args:
            s (str): text.

        Returns:
            (list): tokens.
        """
        self.tokenizer.src_lang = lang
        return self.tokenizer.encode(s)


    def decode(self, tok):
        """
        Args:
            tok (list): list of tokens.

        Returns:
            (str): decoded text.
        """
        return self.tokenizer.decode(tok)


    def convert_ids_to_tokens(self, tok_id):
        """
        Args:
            tok_id (int): token.

        Returns:
            (str): text token.
        """
        return self.tokenizer.convert_ids_to_tokens(tok_id)
    

    def convert_tokens_to_ids(self, tok_str):
        """
        Args:
            tok_str (str): text token.

        Returns:
            (int): token.
        """
        return self.tokenizer.convert_tokens_to_ids(tok_str)
    

    def __len__(self):
        return self.vocab_size