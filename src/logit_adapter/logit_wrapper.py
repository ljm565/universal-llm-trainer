from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_func import LoroLoss



class LogitWrapper(nn.Module):
    def __init__(self, config, base_model, router):
        super(LogitWrapper, self).__init__()
        self.base_model = base_model
        self.router_size = config.router_size
        self.hidden_dim = self.base_model.model.config.hidden_size
        self.vocab_size = config.vocab_size
        self.router = router(
            r=config.r,
            dropout=config.dropout,
            in_features=self.hidden_dim,
            out_features=self.router_size
        )
        self.lm_heads = nn.ModuleList([nn.Linear(self.hidden_dim, self.vocab_size, bias=False) for _ in range(self.router_size-1)])   # Remove one lm_head because of the original model's lm_head

        # Freeze pre-trained base model
        self._freeze_base_model()
        self._init_lm_heads()

        # Init etc
        self.device = self.base_model.device
        self.tokenizer = self.base_model.tokenizer
        self.criterion = LoroLoss(self.tokenizer.pad_token_id)
    
    
    def _freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    
    def _init_lm_heads(self):
        with torch.no_grad():
            for lm_head in self.lm_heads:
                lm_head.weight.copy_(self.base_model.model.lm_head.weight)
                try:
                    lm_head.bias.copy_(self.base_model.model.lm_head.bias)
                # No bias case
                except:
                    continue


    def masking(self, last_hidden_state, router_attention_mask=None):
        if router_attention_mask != None:
            last_hidden_state = last_hidden_state * router_attention_mask.unsqueeze(-1)
        return last_hidden_state
    
    
    @staticmethod
    def pooling(router_values, mode, router_attention_mask=None):
        if mode == 'max':
            router_values = torch.max(router_values, dim=1, keepdim=True)[0]
            router_values = F.softmax(router_values, dim=-1)    # Calcuate router weights
        elif mode == 'avg':
            if router_attention_mask == None:
                router_values = torch.mean(router_values, dim=1, keepdim=True)
            else:
                router_values = torch.sum(router_values, dim=1, keepdim=True) / torch.sum(router_attention_mask.unsqueeze(-1), dim=1, keepdim=True)
            router_values = F.softmax(router_values, dim=-1)    # Calcuate router weights
        else:
            router_values = F.softmax(router_values, dim=-1)    # Calcuate router weights

        return router_values


    def forward(self,
                batch,
                router_attention_mask=None,
                pooling: Optional[Union[str, None]] = 'avg',
                return_loss=False,
        ):
        """Router network forwarding

        Args:
            batch (torch.tensor): batch_data
            router_attention_mask (torch.tensor, optional): (batch x sequence_len) size of tensor. Last_hidden_statates will be masked where the value is 0. Defaults to None.
            pooling (str, optional): [max, avg] or None. If None, we don't operate sequence-wise pooling. Defaults to avg.
            return_loss (bool, optional): Whether return loss value or not. Defaults to False.
        """
        # Forward base_model
        output, orig_loss = self.base_model(batch, return_loss=True, output_hidden_states=True)

        # Router weight calculation
        last_hidden_state = output.hidden_states[-1]
        router_wts = self.router(self.masking(last_hidden_state, router_attention_mask))      # (batch x sequence_len x router_size)
        router_wts = self.pooling(router_wts, pooling, router_attention_mask)       # (batch x 1 x router_size) or (batch x seq_len x router_size)
        
        # Weighted sum
        logits = [output.logits] + [lm_head(last_hidden_state) for lm_head in self.lm_heads]
        logits = sum(router_wts[..., i:i+1] * logits[i] for i in range(len(logits)))

        if return_loss:
            loss = self.criterion(
                logits=logits,
                label=batch['label'],
                router_wts=router_wts,
                router_label=batch['router_label'],
            )
            loss['orig_loss'] = orig_loss
            return logits, router_wts, loss

        return logits, router_wts


    def inference(self, src, max_length, num_return_sequences=1, greedy=False, max_time=None, synced_gpus=False, pooling='avg'):
        if isinstance(src, str):
            src_tok = torch.tensor(self.tokenizer.encode(src), dtype=torch.long).unsqueeze(0).to(self.device)
            if src_tok.size(1) >= max_length:
                return src
            return self.tokenizer.decode(self.generate(src_tok, max_length, num_return_sequences, greedy, max_time, synced_gpus, pooling)[0][src_tok.size(-1):].tolist())
        elif isinstance(src, list):
            assert all([isinstance(s, str) for s in src]), f'All elements in src should be str type'
            src_tok = [torch.tensor(self.tokenizer.encode(s), dtype=torch.long).unsqueeze(0).to(self.device) for s in src]
            return [self.tokenizer.decode(self.generate(tok, max_length, num_return_sequences, greedy, max_time, synced_gpus, pooling)[0][tok.size(-1):].tolist()) if tok.size(1) < max_length else src[i] for i, tok in enumerate(src_tok)]
        else:
            raise AssertionError('Inference input should be str or list of str')
        
    
    def generate(self, src_tok, max_length, num_return_sequences=1, greedy=False, max_time=None, synced_gpus=False, pooling='avg'):
        # Inference original model
        def _inference_base_model(src_tok):
            attention_mask = torch.ones_like(src_tok).to(self.device)
            output = self.base_model.model(
                input_ids=src_tok,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return output.logits, output.hidden_states[-1]
        
        router_attention_mask = torch.ones_like(src_tok).to(self.device)
        for _ in range(max_length):
            # Calculate base model's logits and hidden_states
            logits, last_hidden_state = _inference_base_model(src_tok)
            
            # Router weight calculation
            router_wts = self.router(self.masking(last_hidden_state, router_attention_mask))      # (batch x sequence_len x router_size)
            router_wts = self.pooling(router_wts, pooling, router_attention_mask)       # (batch x 1 x router_size) or (batch x seq_len x router_size)

            # Weighted sum
            logits = [logits] + [lm_head(last_hidden_state) for lm_head in self.lm_heads]
            logits = sum(router_wts[..., i:i+1] * logits[i] for i in range(len(logits)))

            # Predict next token
            src_tok = torch.cat((src_tok, torch.argmax(logits[:, -1], dim=-1).unsqueeze(1)), dim=1)
            
            # Router mask zero padding
            new_mask = torch.zeros((router_attention_mask.size(0), 1), dtype=router_attention_mask.dtype).to(self.device)  # (batch x 1)
            router_attention_mask = torch.cat((router_attention_mask, new_mask), dim=1)

            if src_tok[0, -1].item() in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]:
                break
        
        return src_tok[0].tolist()



