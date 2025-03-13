import torch
import torch.nn.functional as F



def logits_to_logprob(logits:torch.Tensor,
                      labels:torch.Tensor,
                      temperature:float=1.0) -> torch.Tensor:
    """
    Calculate logits' log probabilities.

    Args:
        logits (torch.Tensor): The logits tensor. Shape: [batch, sequence_length, vocab_size]
        labels (torch.Tensor): The label tensor. Shape: [batch, sequence_length]
        temperature (float, optional): The temperature to scale the logits. Defaults to 1.0.

    Returns:
        torch.Tensor: Gathered logits' log probabilities.
    """
    log_probs = F.log_softmax(logits[:, :-1, :] / temperature, dim=-1)
    return torch.gather(log_probs, 2, labels[:, 1:].unsqueeze(-1)).squeeze(-1)



def get_logprob(logits:torch.Tensor,
                labels:torch.Tensor,
                mask:torch.Tensor,
                temperature:float=1.0,
                return_average:bool=False) -> torch.Tensor:
    """
    Calculate average (or sum) values of logits' log probabilities with a mask.

    Args:
        logits (torch.Tensor): The logits tensor. Shape: [batch, sequence_length, vocab_size]
        labels (torch.Tensor): The label tensor. Shape: [batch, sequence_length]
        mask (torch.Tensor): The mask tensor. Shape: [batch, sequence_length]. Defaults to None.
        temperature (float, optional): The temperature to scale the logits. Defaults to 1.0.
        return_average (bool, optional): Whether return average or sum values. Defaults to False.

    Returns:
        torch.Tensor: An average (or sum) of logits' log probabilities applied with a mask.
    """
    per_token_logprobs = logits_to_logprob(logits, labels, temperature)
    
    # Masking
    mask = mask[:, 1:]
    per_token_logprobs = per_token_logprobs * mask
    
    if return_average:
        return per_token_logprobs.sum(-1) / (mask.sum(-1) + 1e-8)

    return per_token_logprobs.sum(-1)