from .abstract_trainer import AbstractTrainer
from ..algorithms.linear_bandits import LinearBandit
import torch


class LinearTrainer(AbstractTrainer[LinearBandit]):
    def __init__(self) -> None:
        pass

    def update(
        self,
        bandit: LinearBandit,
        rewards: torch.Tensor, # shape: (batch_size,)
        chosen_actions: torch.Tensor, # shape: (batch_size, features)
    ) -> LinearBandit:
        """Perform an update"""
        
        # Update the bandit
        bandit.M += chosen_actions.T @ chosen_actions # shape: (features, features)
        bandit.b += chosen_actions.T @ rewards # shape: (features,)
        bandit.theta = torch.linalg.solve(bandit.M, bandit.b) # shape: (features,)
        
        return bandit
        