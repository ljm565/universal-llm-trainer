import torch
import torch.nn as nn
import torch.nn.functional as F



class LoroLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(LoroLoss, self).__init__()
        self.logits_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.router_criterion = nn.NLLLoss()


    def calculate_logits_loss(self, logits, label):
        return self.logits_criterion(logits[:, :-1, :].reshape(-1, logits.size(-1)), label[:, 1:].reshape(-1))
    

    def calculate_router_loss(self, router_wts, router_label):
        # ['avg', 'max'] pooling cases
        # router_wts dim: (batch, 1, router_size)
        # router_label dim: (batch, )
        if router_wts.size(1) == 1:
            router_wts = router_wts.squeeze(1)
        
        # Not pooled case
        # router_wts dim: (batch, seq, router_size)
        # router_label dim: (batch, )
        else:
            router_label = router_label.unsqueeze(1).expand(router_label.size(0), router_wts.size(1))   # (batch, seq)
            router_wts, router_label = router_wts.reshape(-1, router_wts.size(-1)), router_label.reshape(-1)

        return self.router_criterion(torch.log(router_wts), router_label)   # Same as the cross entropy loss because router_wts is already softmax output


    def forward(self, logits, label, router_wts, router_label):
        logits_loss = self.calculate_logits_loss(logits, label)
        router_loss = self.calculate_router_loss(router_wts, router_label)
        loss = logits_loss + router_loss * 1.5
        return {'loss': loss, 'logits_loss': logits_loss, 'router_loss': router_loss}