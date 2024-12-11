from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F



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
    
    
    def _freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False


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
        output = self.base_model(batch, return_loss, output_hidden_states=True)

        # Router weight calculation
        last_hidden_state = self.masking(last_hidden_state, router_attention_mask)
        router_wts = self.router(last_hidden_state)      # (batch x sequence_len x router_size)
        router_wts = self.pooling(router_wts, pooling, router_attention_mask)       # (batch x 1 x router_size) or (batch x seq_len x router_size)
        
        # Weighted sum
        logits = [original_logits] + [lm_head(last_hidden_state) for lm_head in self.lm_heads]
        logits = sum(router_wts[..., i:i+1] * logits[i] for i in range(len(logits)))

        return logits, router_wts




if __name__ == '__main__':
    hidden_dim = 5
    vocab_size = 10
    router_size = 3

    original_logits = torch.randn(2, 6, vocab_size)
    last_hidden_state = torch.randn(2, 6, hidden_dim)
    router_attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    pooling = None
    
    model = LogitWrapper(hidden_dim, vocab_size, router_size)
    output = model(
        original_logits,
        last_hidden_state,
        router_attention_mask,
        pooling
    )







