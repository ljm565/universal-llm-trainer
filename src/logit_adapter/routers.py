import math

import torch.nn as nn



class LoraRouter(nn.Module):
    def __init__(self, r, dropout, in_features, out_features):
        super(LoraRouter, self).__init__()
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.activation = nn.SiLU()
        self.dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # self.layernorm = nn.LayerNorm(in_features)
            

    def _reset_params(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)


    def forward(self, last_hidden_state):
        # last_hidden_state = self.lora_B(self.activation(self.lora_A(self.dropout_layer(last_hidden_state))))
        # last_hidden_state = self.layernorm(last_hidden_state)
        last_hidden_state = self.lora_B(self.lora_A(self.dropout_layer(last_hidden_state)))
        return last_hidden_state