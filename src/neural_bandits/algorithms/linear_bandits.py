import torch

from .abstract_bandit import AbstractBandit


class LinearBandit(AbstractBandit):
    def __init__(self, n_arms: int, n_features: int) -> None:
        super().__init__(n_arms, n_features)

        self.M: torch.Tensor = torch.eye(n_features)
        self.b = torch.zeros(n_features)
        self.theta = torch.zeros(n_features)
        


class LinearTSBandit(LinearBandit):
    def __init__(self, n_arms: int, n_features: int) -> None:
        super().__init__(n_arms, n_features)
        
    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        assert contextualised_actions.shape[1] == self.n_arms and contextualised_actions.shape[2] == self.n_features, "Contextualised actions must have shape (batch_size, n_arms, n_features)"
        batch_size = contextualised_actions.shape[0]
        
        theta_tilde = torch.distributions.MultivariateNormal(self.theta, torch.inverse(self.M)).sample((batch_size,))  # type: ignore
        
        return torch.argmax(torch.einsum("ijk,ik->ij", contextualised_actions, theta_tilde), dim=1) # TODO


class LinearUCBBandit(LinearBandit):
    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0) -> None:
        super().__init__(n_arms, n_features)
        self.alpha = alpha

    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        assert contextualised_actions.shape[1] == self.n_arms and contextualised_actions.shape[2] == self.n_features, "Contextualised actions must have shape (batch_size, n_arms, n_features)"
        
        M_inv = torch.inverse(self.M)
        
        return torch.argmax(
            torch.einsum("ijk,k->ij", contextualised_actions, self.theta)
            + self.alpha
            * torch.sqrt(
                torch.einsum("ijk,kl,ijl->ij", contextualised_actions, M_inv, contextualised_actions)
            )
        ) # TODO

class LinearTSApproxBandit(LinearTSBandit):  # TODO
    pass


class LinearUCBApproxBandit(LinearUCBBandit):  # TODO
    pass

