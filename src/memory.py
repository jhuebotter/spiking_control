from typing import Any
import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass, field


@dataclass
class Transition:
    state: Tensor
    target: Tensor
    action: Tensor
    reward: float
    done: bool
    next_state: Tensor

    def __getitem__(self, key: str) -> Tensor:
        return torch.tensor(getattr(self, key))
    
    def as_dict(self) -> dict[str, Tensor]:
        return {
            'state': self.state,
            'target': self.target,
            'action': self.action,
            'reward': torch.tensor(self.reward),
            'done': torch.tensor(self.done),
            'next_state': self.next_state
        }


@dataclass
class Episode:
    steps_: list[Transition] = field(default_factory=list)

    def append(self, transition: Transition) -> None:
        self.steps_.append(transition)

    def __len__(self) -> int:
        return len(self.steps_)
    
    def __getitem__(self, index: int) -> Transition:
        return self.steps_[index]


class BaseMemory:
    """buffer object with maximum size to store recent experience"""
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.reset()

    def reset(self) -> None:
        self.buffer = [None] * self.max_size
        self.index = 0
        self.size = 0

    def append(self, obj: object) -> None:
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size: int) -> list:
        indices = np.random.choice(range(self.size), batch_size)
        
        return [self.buffer[index] for index in indices]


class EpisodeMemory(BaseMemory):
    """buffer object with maximum size to store recent experience"""
    def __init__(self, max_size: int) -> None:
        super().__init__(
            max_size=max_size
        )

    def sample_batch(
        self,
        batch_size: int,
        warmup_steps: int,
        unroll_steps: int,
        device: torch.device,
    ) -> list[Tensor]:
        """sample a batch of episodes from memory and return a batch of state, target, action, reward, next_state tensors
            of shape [warmup_steps + unroll_steps, batch_size, dim]

        Args:
            batch_size (int): number of episodes to sample
            warmup_steps (int): number of steps to warm up the models
            unroll_steps (int): number of steps to unroll the models
            device (torch.device): device to store tensors on

        Returns:
            list[Tensor]: list of tensors of shape [warmup_steps + unroll_steps, batch_size, dim]
        """

        steps = warmup_steps + unroll_steps
        assert 0 < steps
        assert 0 < self.size
        episodes = []

        # TODO: need to make sure that episodes are not too short

        # sample episodes until batch is full
        while len(episodes) < batch_size:
            sample_batch = self.sample(
                batch_size - len(episodes)
            )  # [sample, step, (state, target, action, reward, done, next_state)]
            for episode in sample_batch:
                if len(episode) >= steps:
                    episodes.append(episode)
                    if len(episodes) == batch_size:
                        break

        # make a random sample from each episode of length = warmup_steps
        state_dim, target_dim, action_dim, reward_dim, done_dim, next_state_dim = (
            episodes[0][0]['state'].size(-1),
            episodes[0][0]['target'].size(-1),
            episodes[0][0]['action'].size(-1),
            1,
            1,
            episodes[0][0]['state'].size(-1),
        )

        # initialize tensors
        state_batch = torch.zeros((steps, batch_size, state_dim), device=device)
        target_batch = torch.zeros((steps, batch_size, target_dim), device=device)
        action_batch = torch.zeros((steps, batch_size, action_dim), device=device)
        reward_batch = torch.zeros((steps, batch_size, reward_dim), device=device)
        done_batch = torch.zeros((steps, batch_size, done_dim), device=device)
        next_state_batch = torch.zeros((steps, batch_size, next_state_dim), device=device)

        # fill tensors with random samples from episodes
        for j, episode in enumerate(episodes):
            r = torch.randint(low=0, high=len(episode) - steps, size=(1,))
            state_batch[:, j, :] = torch.stack(
                [step['state'].squeeze() for step in episode[r : r + steps]]
            )
            target_batch[:, j, :] = torch.stack(
                [step['target'].squeeze() for step in episode[r : r + steps]]
            )
            action_batch[:, j, :] = torch.stack(
                [step['action'].squeeze() for step in episode[r : r + steps]]
            )
            reward_batch[:, j, :] = torch.stack(
                [step['reward'].unsqueeze_(-1) for step in episode[r : r + steps]]
            )
            done_batch[:, j, :] = torch.stack(
                [step['done'].unsqueeze_(-1) for step in episode[r : r + steps]]
            )
            next_state_batch[:, j, :] = torch.stack(
                [step['next_state'].squeeze() for step in episode[r : r + steps]]
            )

        return [state_batch, target_batch, action_batch, reward_batch, done_batch, next_state_batch]
