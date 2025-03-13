import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Output



class RSOLoss(nn.Module):
    def __init__(self, beta:float=0.1):
        super(RSOLoss, self).__init__()
        """
        Statistical Rejection Sampling Optimization (RSO) loss calculation class.

        Args:
            beta (float): Equivalent temperature parameter (from DPO) for the RSO loss.

        Notes:
            - https://arxiv.org/abs/2309.06657
        """
        self.beta = beta


    def forward(self,
                policy_chosen_logp: torch.Tensor,
                policy_rejected_logp: torch.Tensor,
                reference_chosen_logp: torch.Tensor,
                reference_rejected_logp: torch.Tensor) -> Output:
        """
        Calculate RSO loss and other metrics.

        Args:
            policy_chosen_logp (torch.Tensor): Preferred log probabilities from being trained model.
            policy_rejected_logp (torch.Tensor): Non-preferred log probabilities from being trained model.
            reference_chosen_logp (torch.Tensor): Preferred log probabilities from reference model.
            reference_rejected_logp (torch.Tensor): Non-preferred log probabilities from reference model.

        Returns:
            Output: Loss, rewards, and other reward related metrics.
        """
        pi_logratios = policy_chosen_logp - policy_rejected_logp
        ref_logratios = reference_chosen_logp - reference_rejected_logp

        # Loss calculation
        logits = pi_logratios - ref_logratios
        losses = torch.relu(1 - self.beta * logits)

        # Metric calculation
        chosen_rewards = self.beta * (policy_chosen_logp - reference_chosen_logp).detach()
        rejected_rewards = self.beta * (policy_rejected_logp - reference_rejected_logp).detach()
        reward_acc = (chosen_rewards > rejected_rewards).float().mean(dim=-1)
        reward_margins = (chosen_rewards - rejected_rewards).mean(-1)

        output = Output(
            loss=losses,
            chosen_reward=chosen_rewards,
            rejected_reward=rejected_rewards,
            reward_acc=reward_acc,
            reward_margin=reward_margins,
        )
    
        return output
