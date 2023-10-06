from . import BaseAgent
from ..memory import EpisodeMemory, Transition, Episode
from ..models import make_transition_model, make_policy_model
from ..utils import make_optimizer

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional


class PredictiveControlAgent(BaseAgent):
    def __init__(
            self, 
            env: gym.vector.VectorEnv, 
            config: DictConfig,
            device: torch.device
            ):
        
        super().__init__(
            env, 
            config,
            device
            )

        # initialize config
        self.memory_config = config.agent.memory
        self.transition_config = config.transition
        self.policy_config = config.policy

        # training parameters
        self.reset_memory = config.agent.reset_memory
        self.steps_per_iteration = config.agent.steps_per_iteration
        self.transition_batches_per_iteration = self.transition_config.learning.params.get("batches_per_iteration", 1)
        self.transition_batch_size = self.transition_config.learning.params.get("batch_size", 128)
        self.policy_batches_per_iteration = self.policy_config.learning.params.get("batches_per_iteration", 1)
        self.policy_batch_size = self.policy_config.learning.params.get("batch_size", 128)

        # initialize dimensions
        self.action_dim = env.action_space.shape[1]
        self.state_dim = env.observation_space['proprio'].shape[1]
        self.target_dim = env.observation_space['target'].shape[1]

        self.setup()

    def setup(self):

        # initialize memory
        self.memory = EpisodeMemory(**self.memory_config.params)

        # initialize models
        self.transition_model = make_transition_model(
            action_dim = self.action_dim, 
            state_dim = self.state_dim, 
            config = self.transition_config.get("model", {})
        ).to(self.device)

        self.policy_model = make_policy_model(
            action_dim = self.action_dim,
            state_dim = self.state_dim,
            target_dim = self.target_dim,
            config = self.policy_config.get("model", {})
        ).to(self.device)

        # initialize optimizers
        self.transition_model.set_optimizer(make_optimizer(
            self.transition_model, 
            self.transition_config.get("optimizer", {})
        ))

        self.policy_model.set_optimizer(make_optimizer(
            self.policy_model, 
            self.policy_config.get("optimizer", {})
        ))

        # initialize counters
        self.steps = 0
        self.iterations = 0
        self.episodes = 0
        self.policy_updates = 0
        self.transition_updates = 0

    def run(self, total_steps: int):

        while self.steps < total_steps:
            self.collect_rollouts(self.steps_per_iteration, self.reset_memory)
            self.train()
            self.test()
            self.steps += self.steps_per_iteration
            self.iterations += 1

    def collect_rollouts(self, steps: int, reset_memory: bool = True):
        
        self.policy_model.eval()

        self.memory.reset() if reset_memory else None

        step = 0
        num_envs = self.env.num_envs

        action_min = torch.tensor(self.env.action_space.low, device=self.device)
        action_max = torch.tensor(self.env.action_space.high, device=self.device)

        with torch.no_grad():

            # reset the environment
            observations, infos = self.env.reset()
            obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
            targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)

            self.policy_model.reset_state()

            episodes = [Episode() for _ in range(num_envs)]
            
            with tqdm(
                total=steps,
                desc=f"{'obtaining experience':30}",
            ) as pbar:
                while step < steps:

                    # predict the action
                    actions = self.policy_model(obs, targets)
                    actions = actions.squeeze(0).clamp(action_min, action_max).detach()

                    # step the environment
                    observations, rewards, terminates, truncateds, infos = self.env.step(actions.cpu().numpy())

                    dones = [True if ter or tru else False for ter, tru in zip(terminates, truncateds)]
                    next_obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                    #next_targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)

                    if '_final_observation' in infos.keys():
                        final = infos['_final_observation']
                        for i, f in enumerate(final):
                            if f:
                                next_obs[i] = torch.tensor(infos['final_observation'][i]['proprio'], device=self.device, dtype=torch.float32)
                                #next_targets[i] = torch.tensor(infos['final_observation'][i]['target'], device=self.device, dtype=torch.float32)

                    # store the transition
                    for i, (o, t, a, r, d, no) in enumerate(zip(obs, targets, actions, rewards, dones, next_obs)):
                        episodes[i].append(Transition(o, t, a, r, d, no))
                        if d:
                            # ! This only adds complete episodes to the memory
                            self.memory.append(episodes[i])
                            episodes[i] = Episode()
                            self.episodes += 1

                    # update the state
                    obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                    targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)

                    step += num_envs
                    pbar.update(num_envs)

    def train_epoch(self):

        transition_results = self.train_transition_model()
        policy_results = self.train_policy_model()

    def train_transition_model(self):

        # train the transition model
        transition_results = []
        n_transition_batches = self.transition_config.learning.params.get("batches_per_iteration", 1)
        pbar = tqdm(range(n_transition_batches), desc=f"{'training transition model':30}")
        for batch in pbar:
            transition_result = self.transition_model.train_fn(
                self.memory,
                self.transition_batch_size,
                **self.transition_config.get("train_fn", {})
            )
            self.transition_updates += 1
            loss = transition_result['transition model loss']
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            transition_results.append(transition_result)

        return transition_results

    def train_policy_model(self):
        
        # train the policy model
        policy_results = []
        n_policy_batches = self.policy_config.learning.params.get("batches_per_iteration", 1)
        pbar = tqdm(range(n_policy_batches), desc=f"{'training policy model':30}")
        for batch in pbar:
            policy_result = self.policy_model.train_fn(
                self.memory,
                self.transition_model,
                self.env.call('get_loss_gain')[0],
                self.policy_batch_size,
                **self.policy_config.get("train_fn", {})
            )
            self.policy_updates += 1
            loss = policy_result['policy model loss']
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            policy_results.append(policy_result)

        return policy_results

    def train(self, epochs: int = 1):
        
        for e in range(epochs):
            self.train_epoch()

    def test(self, env: Optional[gym.vector.VectorEnv] = None, episodes: int = 1, render: bool = False):
        
        if env is None:
            env = self.env

        self.policy_model.eval()

        with torch.no_grad():
                
                for e in range(episodes):
                    observations, infos = env.reset()
                    obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                    targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)
    
                    self.policy_model.reset_state()
    
                    total_reward = 0
                    done = False
                    while not done:
                        if render:
                            env.render()
    
                        action = self.policy_model(obs, targets)
                        action = action.squeeze(0).clamp(-1, 1).detach()
    
                        observations, rewards, terminates, truncateds, infos = env.step(action.cpu().numpy())
                        obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                        targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)
    
                        if '_final_observation' in infos.keys():
                            final = infos['_final_observation']
                            for i, f in enumerate(final):
                                if f:
                                    obs[i] = torch.tensor(infos['final_observation'][i]['proprio'], device=self.device, dtype=torch.float32)
                                    targets[i] = torch.tensor(infos['final_observation'][i]['target'], device=self.device, dtype=torch.float32)
    
                        total_reward += sum(rewards) / env.num_envs
                        done = True if terminates[0] or truncateds[0] else False

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError