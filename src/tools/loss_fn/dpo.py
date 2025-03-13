import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Output



class DPOLoss(nn.Module):
    def __init__(self, beta:float=0.1, label_smoothing:float=0.0):
        """
        Direct Preference Optimization (DPO) loss calcuation class.

        Args:
            beta (float, optional):  A scaling parameter that adjusts the sensitivity of the loss function to the 
                                     differences between the policy and reference models. A higher beta increases 
                                     the emphasis on the difference, leading to sharper updates in policy optimization.
                                     A smaller beta smooths out the differences, making learning more stable but slower.
                                     Typical values range between 0.1 and 0.5. Defaults to 0.1.
            label_smoothing (float, optional): A regularization parameter that prevents the model from being overly confident 
                                               in its predictions. It adjusts the loss function by blending positive and negative
                                               signals during optimization. When label_smoothing=0, the loss function behaves 
                                               traditionally, strongly favoring correct predictions. As label_smoothing increases, 
                                               the loss becomes more balanced, accounting for uncertainty in the labels. Defaults to 0.0.
        
        Notes:
            - https://arxiv.org/abs/2305.18290
        """
        super(DPOLoss, self).__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing


    def forward(self,
                policy_chosen_logp: torch.Tensor,
                policy_rejected_logp: torch.Tensor,
                reference_chosen_logp: torch.Tensor,
                reference_rejected_logp: torch.Tensor) -> Output:
        """
        Calculate DPO loss and other metrics.

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
        losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing

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
