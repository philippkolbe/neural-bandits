from typing import Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset


class WheelBanditDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Generates a dataset for the Wheel Bandit problem (see https://arxiv.org/abs/1802.09127).
    For this implementation contexts have d=2 dimensions and k=5 actions.
    Contexts are sampled uniformally. Rewards are sampled according to the following rules:
        - If the context norm is below delta, the reward is sampled from a Gaussian distribution with mean mean_v and std std_v (per action).
        - If the context norm is above delta, the reward is sampled from a Gaussian distribution with mean mu_large and std std_large for the action that fits to the context (one action per region).

    The default parameters from the original paper are:
        - mu_0 = 1.2
        - mu_1 = mu_2 = mu_3 = mu_4 = 1.0
        - mu_large = 50.0
        - std_large = std_0 = std_1 = std_2 = std_3 = std_4 = 0.01

    Args:
        num_contexts: Number of points to sample, i.e. (context, action rewards).
        delta: Exploration parameter: high reward in one region if norm above delta.
        mean_v: Mean reward for each action if context norm is below delta.
        std_v: Gaussian reward std for each action if context norm is below delta.
        mu_large: Mean reward for optimal action if context norm is above delta.
        std_large: Reward std for optimal action if context norm is above delta.
        seed: Random seed for reproducibility.
    """

    num_actions: int = 5
    context_dim: int = 2

    def __init__(
        self,
        num_samples: int,
        delta: float,
        mean_v: NDArray[np.float32] = np.array([1.2, 1.0, 1.0, 1.0, 1.0]),
        std_v: NDArray[np.float32] = np.array([0.01, 0.01, 0.01, 0.01, 0.01]),
        mu_large: float = 50.0,
        std_large: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.delta = delta
        self.mean_v = mean_v
        self.std_v = std_v
        self.mu_large = mu_large
        self.std_large = std_large

        self.seed = seed
        data, opt = self._generate_data()
        self.data = data
        self.opt = opt

    def _generate_data(
        self,
    ) -> Tuple[NDArray[np.float32], Tuple[NDArray[np.float32], NDArray[np.int64]]]:
        """Implementation adapted from https://github.com/mlisicki/deep_contextual_bandits/blob/master/bandits/data/synthetic_data_sampler.py.

        Args:
            num_contexts: Number of points to sample, i.e. (context, action rewards).
            delta (\in (0, 1)): Exploration parameter: high reward in one region if norm above delta.
            mean_v: Mean reward for each action if context norm is below delta.
            std_v: Gaussian reward std for each action if context norm is below delta.
            mu_large: Mean reward for optimal action if context norm is above delta.
            std_large: Reward std for optimal action if context norm is above delta.

        Returns:
            dataset: Sampled matrix with n rows: (context, action rewards).
            opt_vals: Vector of expected optimal (reward, action) for each context.
        """

        data: list[NDArray[np.float32]] = []
        rewards = []
        opt_actions: list[np.int64] = []
        opt_rewards: list[np.float32] = []

        # sample uniform contexts in unit ball
        while len(data) < self.num_samples:
            raw_data = np.random.uniform(
                -1, 1, (int(self.num_samples / 3), self.context_dim)
            ).astype(np.float32)

            for i in range(raw_data.shape[0]):
                if np.linalg.norm(raw_data[i, :]) <= 1:
                    data.append(raw_data[i, :])

        contexts = np.stack(data)[: self.num_samples, :]

        # sample rewards
        for i in range(self.num_samples):
            r = [
                np.random.normal(self.mean_v[j], self.std_v[j])
                for j in range(self.num_actions)
            ]
            if np.linalg.norm(contexts[i, :]) > self.delta:
                # large reward in the right region for the context
                r_big = np.random.normal(self.mu_large, self.std_large)
                if contexts[i, 0] > 0:
                    if contexts[i, 1] > 0:
                        r[0] = r_big
                        opt_actions.append(np.int64(0))
                    else:
                        r[1] = r_big
                        opt_actions.append(np.int64(1))
                else:
                    if contexts[i, 1] > 0:
                        r[2] = r_big
                        opt_actions.append(np.int64(2))
                    else:
                        r[3] = r_big
                        opt_actions.append(np.int64(3))
            else:
                opt_actions.append(np.argmax(self.mean_v))

            opt_rewards.append(r[opt_actions[-1]])
            rewards.append(r)

        rewards = np.stack(rewards)

        return np.hstack((contexts, rewards)), (
            np.array(opt_rewards, dtype=np.float32),
            np.array(opt_actions, dtype=np.int64),
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, reward = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            reward, dtype=torch.float32
        )
