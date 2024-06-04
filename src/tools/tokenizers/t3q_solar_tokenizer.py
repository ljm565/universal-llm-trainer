from transformers import AutoTokenizer



class T3QSolarTokenizer:
    def __init__(self, config, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.remapping_special_tokens(config)
        
        # special tokens
        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.bos_token, self.bos_token_id = self.tokenizer.bos_token, self.tokenizer.bos_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.eos_token, self.tokenizer.eos_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id
        self.cls_token, self.cls_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.sep_token, self.sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id

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


    def encode(self, s):
        """
        Args:
            s (str): text.

        Returns:
            (list): tokens.
        """
        # # solar tokenizer attatches "_" (empty space) token in front of the Korean text
        tokens = self.tokenizer.encode(s, add_special_tokens=False)
        # if tokens[0] == 28705:
        #     return tokens[1:]
        return tokens


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
    

    def remapping_special_tokens(self, config):
        new_map = {
            'pad_token_id': config.pad_token_id if isinstance(config.pad_token_id, int) else None,
            'bos_token_id': config.bos_token_id if isinstance(config.bos_token_id, int) else None,
            'eos_token_id': config.eos_token_id if isinstance(config.eos_token_id, int) else None,
            'cls_token_id': config.cls_token_id if isinstance(config.cls_token_id, int) else None,
            'sep_token_id': config.sep_token_id if isinstance(config.sep_token_id, int) else None,
        }

        # check sanity of special token ids
        for k, v in new_map.items():
            if v is not None and v > len(self.tokenizer):
                raise ValueError(f'Invalid special token id: {k}={v} > vocab size={len(self.tokenizer)}')
    
        # remapping special tokens, not add tokens
        for k, v in new_map.items():
            if v is not None:
                if hasattr(self.tokenizer, k):
                    setattr(self.tokenizer, k, v)
                else:
                    raise AttributeError(f'No attribute: {k} in tokenizer. Please check the attribute name.')
                
        # add special tokens
        add_tokens = {
            'pad_token_id': config.pad_token_id if config.pad_token_id == 'add' else None,
            'bos_token_id': config.bos_token_id if config.bos_token_id == 'add' else None,
            'eos_token_id': config.eos_token_id if config.eos_token_id == 'add' else None,
            'cls_token_id': config.cls_token_id if config.cls_token_id == 'add' else None,
            'sep_token_id': config.sep_token_id if config.sep_token_id == 'add' else None,
        }
        
        for k, v in add_tokens.items():
            if v is not None:
                token = f"<{k.split('_')[0]}>"
                self.tokenizer.add_special_tokens({f'{k[:-3]}': token})
                self.resized = True