from . import BaseAgent
from ..memory import EpisodeMemory, Transition, Episode
from ..models import (
    make_transition_model,
    make_policy_model
)
from ..utils import (
    make_optimizer,
    conf_to_dict,
    dict_mean,
    FrameStack
)

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional
from pathlib import Path


class PredictiveControlAgent(BaseAgent):
    def __init__(
            self, 
            env: gym.vector.VectorEnv, 
            config: DictConfig,
            device: torch.device,
            loggers: list = [],
            dir: Optional[str] = None,
            eval_env: Optional[gym.vector.VectorEnv] = None
            ):
        
        super().__init__(
            env, 
            config,
            device,
            loggers,
            dir,
            eval_env
            )

        # initialize config
        self.memory_config = config.agent.memory
        self.transition_config = config.transition
        self.policy_config = config.policy
        self.run_config = config.run

        # training parameters
        self.reset_memory = config.agent.reset_memory
        self.steps_per_iteration = config.agent.steps_per_iteration
        self.steps_per_evaluation = config.agent.steps_per_evaluation
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
            config = conf_to_dict(self.transition_config.get("model", {}))
        ).to(self.device)

        self.policy_model = make_policy_model(
            action_dim = self.action_dim,
            state_dim = self.state_dim,
            target_dim = self.target_dim,
            config = conf_to_dict(self.policy_config.get("model", {}))
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
        self.episodes = 0
        self.epochs = 0
        self.iterations = 0
        self.policy_updates = 0
        self.transition_updates = 0

    def run(self, total_steps: int):

        while self.steps < total_steps:
            self.collect_rollouts(self.steps_per_iteration, self.reset_memory)
            self.train()
            if self.iterations % self.run_config.get("render_every", 1) == 0:
                render = True
            else:
                render = False
            self.test(steps=self.steps_per_evaluation, render=render)
            self.save_models()
            self.iterations += 1

        self.finish_run()

    def collect_rollouts(self, steps: int, reset_memory: bool = True):
        
        self.policy_model.eval()

        if reset_memory: self.memory.reset()

        num_envs = self.env.num_envs

        episodes = [Episode() for _ in range(num_envs)]

        action_min = torch.tensor(self.env.action_space.low, device=self.device)
        action_max = torch.tensor(self.env.action_space.high, device=self.device)

        with torch.no_grad():

            # reset the environment
            observations, infos = self.env.reset()
            obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
            targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)

            self.policy_model.reset_state()

            step = 0
            with tqdm(
                total=steps,
                desc=f"{'obtaining experience':30}",
            ) as pbar:
                while step < steps:

                    # predict the action
                    actions = self.policy_model.predict(obs, targets)
                    actions = actions.squeeze(0).clamp(action_min, action_max).detach()

                    # step the environment
                    observations, rewards, terminates, truncateds, infos = self.env.step(actions.cpu().numpy())

                    dones = [True if ter or tru else False for ter, tru in zip(terminates, truncateds)]
                    next_obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                    if '_final_observation' in infos.keys():
                        final = infos['_final_observation']
                        for i, f in enumerate(final):
                            if f:
                                next_obs[i] = torch.tensor(infos['final_observation'][i]['proprio'], device=self.device, dtype=torch.float32)

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

        self.steps += step

    def train_epoch(self):

        # train the models
        transition_results = self.train_transition_model()
        policy_results = self.train_policy_model()

        return transition_results, policy_results

    def train_transition_model(self):

        # train the transition model
        transition_results = []
        n_transition_batches = self.transition_config.learning.get("batches_per_iteration", 1)
        pbar = tqdm(range(n_transition_batches), desc=f"{'training transition model':30}")
        for batch in pbar:
            transition_result = self.transition_model.train_fn(
                self.memory,
                **self.transition_config.get("learning", {}).get("params", {})
            )
            self.transition_updates += 1
            loss = transition_result['transition model loss']
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            transition_results.append(transition_result)

        return transition_results

    def train_policy_model(self):
        
        # train the policy model
        policy_results = []
        n_policy_batches = self.policy_config.learning.get("batches_per_iteration", 1)
        pbar = tqdm(range(n_policy_batches), desc=f"{'training policy model':30}")
        for batch in pbar:
            policy_result = self.policy_model.train_fn(
                self.memory,
                self.transition_model,
                self.env.call('get_loss_gain')[0],
                **self.policy_config.get("learning", {}).get("params", {})
            )
            self.policy_updates += 1
            loss = policy_result['policy model loss']
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            policy_results.append(policy_result)

        return policy_results

    def train(self, epochs: int = 1):
        
        for e in range(epochs):
            transition_results, policy_results = self.train_epoch()

            self.epochs += 1
            
            # collect the training results
            results = {
                "steps": self.steps,
                "episodes": self.episodes,
                "epochs": self.epochs,
                "policy updates": self.policy_updates,
                "transition updates": self.transition_updates,
            }
            results.update(dict_mean(transition_results))
            results.update(dict_mean(policy_results))

            # log the results
            self.log(results, step=self.epochs)

    def test(self, steps: int, env: Optional[gym.vector.VectorEnv] = None, render: bool = False):
        
        if env is None:
            env = self.env if self.eval_env is None else self.eval_env

        num_envs = env.num_envs

        action_min = torch.tensor(env.action_space.low, device=self.device)
        action_max = torch.tensor(env.action_space.high, device=self.device)

        observations, infos = env.reset()
        obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
        targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)

        completed_episodes = []
        episodes = [Episode() for _ in range(num_envs)]

        if render:
            completed_framestacks = []
            framestacks = [FrameStack() for _ in range(num_envs)]
            frames = env.call('render')
            for i, frame in enumerate(frames):
                framestacks[i].append(frame)

        with torch.no_grad(): 
            self.policy_model.eval()       
            self.policy_model.reset_state()

            step = 0
            total_reward = 0
            with tqdm(
                total=steps,
                desc=f"{'evaluating models':30}",
            ) as pbar:
                while step < steps:

                    # predict the action
                    actions = self.policy_model.predict(obs, targets)
                    actions = actions.squeeze(0).clamp(action_min, action_max).detach()

                    # step the environment
                    observations, rewards, terminates, truncateds, infos = env.step(actions.cpu().numpy())

                    dones = [True if ter or tru else False for ter, tru in zip(terminates, truncateds)]
                    next_obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                    if '_final_observation' in infos.keys():
                        final = infos['_final_observation']
                        for i, f in enumerate(final):
                            if f:
                                next_obs[i] = torch.tensor(infos['final_observation'][i]['proprio'], device=self.device, dtype=torch.float32)

                    # store the transition
                    for i, (o, t, a, r, d, no) in enumerate(zip(obs, targets, actions, rewards, dones, next_obs)):
                        episodes[i].append(Transition(o, t, a, r, d, no))
                        if d:
                            completed_episodes.append(episodes[i])
                            total_reward += episodes[i].get_cummulative_reward()
                            episodes[i] = Episode()

                    # store frames for rendering
                    if render:
                        frames = env.call('render')
                        for i, frame in enumerate(frames):
                            if dones[i]:
                                completed_framestacks.append(framestacks[i])
                                framestacks[i] = FrameStack()
                            framestacks[i].append(frame)

                    # update the state
                    obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                    targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)

                    step += num_envs
                    pbar.update(num_envs)

        # make the video
        if render:
            pass
            #make_video(completed_framestacks, self.dir, f"test_{self.epochs}.mp4")

        # log the results
        average_reward = total_reward / len(completed_episodes)


    def save_models(self):

        self.save_transition_model()
        self.save_policy_model()

    def save_transition_model(
            self, 
            dir: Optional[str] = None,
            file: str = "transition_model.cpt", 
            save_optimizer: bool = True
        ):

        if dir is None:
            dir = Path(self.dir, "models")

        self.save(
            model=self.transition_model,
            dir=dir,
            file=file,
            optimizer=self.transition_model.get_optimizer() if save_optimizer else None
        )

    def save_policy_model(            
            self, 
            dir: Optional[str] = None,
            file: str = "policy_model.cpt", 
            save_optimizer: bool = True
        ):

        if dir is None:
            dir = Path(self.dir, "models")

        self.save(
            model=self.policy_model,
            dir=dir,
            file=file,
            optimizer=self.policy_model.get_optimizer() if save_optimizer else None
        )
    
    def load_models(self, dir: Optional[str] = None):

        self.load_transition_model(dir)
        self.load_policy_model(dir)

    def load_transition_model(self, dir: Optional[str] = None, file: str = "transition_model.cpt"):

        if dir is None:
            dir = self.run_config.get("load_path", None)
            assert dir is not None, "No load path specified!"
            dir = Path(dir)
        path = Path(dir, file)

        self.transition_model, op = self.load(
            model=self.transition_model,
            path=path,
            optim=self.transition_model.get_optimizer()
        )
        if op is not None:
            self.transition_model.set_optimizer(op)
    
    def load_policy_model(self, dir: Optional[str] = None, file: str = "policy_model.cpt"):

        if dir is None:
            dir = self.run_config.get("load_path", None)
            assert dir is not None, "No load path specified!"
            dir = Path(dir)
        path = Path(dir, file)

        self.policy_model, op = self.load(
            model=self.policy_model,
            path=path,
            optim=self.policy_model.get_optimizer()
        )
        if op is not None:
            self.policy_model.set_optimizer(op)