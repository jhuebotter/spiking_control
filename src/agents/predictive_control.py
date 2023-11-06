from . import BaseAgent
from src.memory import EpisodeMemory, Transition, Episode
from src.models import (
    make_transition_model,
    make_policy_model
)
from ..utils import (
    make_optimizer,
    dict_mean,
    FrameStack
)
from src.eval_helpers import baseline_prediction

import gymnasium as gym
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Optional
from pathlib import Path
from src.plotting import (
    render_video,
    animate_predictions
)


class PredictiveControlAgent(BaseAgent):
    def __init__(
            self, 
            env: gym.vector.VectorEnv, 
            config: DictConfig,
            device: torch.device,
            loggers: list = [],
            dir: Optional[str] = None,
            eval_env: Optional[gym.vector.VectorEnv] = None,
            id: Optional[str] = None,
            ):
        
        super().__init__(
            env, 
            config,
            device,
            loggers,
            dir,
            eval_env,
            id,
            )

        # initialize config
        self.memory_config = config.agent.memory
        self.transition_config = config.transition
        self.policy_config = config.policy
        self.run_config = config.run
        self.plotting_config = config.plotting

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
            config = self.transition_config.model
        ).to(self.device)

        self.policy_model = make_policy_model(
            action_dim = self.action_dim,
            state_dim = self.state_dim,
            target_dim = self.target_dim,
            config = self.policy_config.model
        ).to(self.device)
        self.models = [self.transition_model, self.policy_model]

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

        self.test(steps=self.steps_per_evaluation, render=True)
        while self.steps < total_steps:
            self.collect_rollouts(self.steps_per_iteration, self.reset_memory)
            self.train()
            render = ((self.iterations + 1) % self.plotting_config.get("render_every", 1)) == 0
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
            total_reward = 0
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
                            total_reward += episodes[i].get_cummulative_reward()
                            episodes[i] = Episode()
                            self.episodes += 1

                    # update the state
                    obs = torch.tensor(observations['proprio'], device=self.device, dtype=torch.float32)
                    targets = torch.tensor(observations['target'], device=self.device, dtype=torch.float32)

                    step += num_envs
                    pbar.update(num_envs)

        self.steps += step
        average_reward = total_reward / len(episodes)
        results = {
            "train average reward": average_reward,
        }

        self.log(results, step=self.epochs)


    def train_transition_model(self):

        # train the transition model
        transition_results = []
        n_transition_batches = self.transition_config.learning.get("batches_per_iteration", 1)
        pbar = tqdm(range(n_transition_batches), desc=f"{'training transition model':30}")
        for batch in pbar:
            transition_result = self.transition_model.train_fn(
                memory = self.memory,
                record = True,
                excluded_monitor_keys = self.transition_model.plot_monitors,
                **self.transition_config.get("learning", {}).get("params", {})
            )
            self.transition_updates += 1
            loss = transition_result['loss']
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
                memory = self.memory,
                transition_model = self.transition_model,
                loss_gain = self.env.call('get_loss_gain')[0],
                record = True,
                excluded_monitor_keys = self.policy_model.plot_monitors,
                **self.policy_config.get("learning", {}).get("params", {})
            )
            self.policy_updates += 1
            loss = policy_result['loss']
            pbar.set_postfix_str(f"loss: {loss:.5f}")
            policy_results.append(policy_result)

        return policy_results
    
    def train_epoch(self):

        # train the models
        transition_results = self.train_transition_model()
        policy_results = self.train_policy_model()

        return transition_results, policy_results

    def train(self, epochs: int = 1):
        
        for e in range(epochs):
            transition_results, policy_results = self.train_epoch()

            self.epochs += 1
            
            # collect the training results
            results = {
                "steps": self.steps,
                "episodes": self.episodes,
                "epochs": self.epochs,
                "policy model updates": self.policy_updates,
                "transition model updates": self.transition_updates,
                "policy model parameters": self.policy_model.count_parameters(),
                "transition model parameters": self.transition_model.count_parameters(),
            }
            results.update(dict_mean(transition_results, prefix=self.transition_model.name + " "))
            results.update(dict_mean(policy_results, prefix=self.policy_model.name + " "))

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
            self.transition_model.eval()
            self.transition_model.reset_state()

            step = 0
            total_reward = 0
            with tqdm(
                total=steps,
                desc=f"{'evaluating models':30}",
            ) as pbar:
                while step < steps:

                    # predict the action
                    actions = self.policy_model.predict(obs, targets, record=True)
                    actions = actions.squeeze(0).clamp(action_min, action_max).detach()

                    # predict the next states
                    _ = self.transition_model.predict(obs, actions, deterministic=True, record=True)

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

        # log the results
        average_reward = total_reward / len(completed_episodes)
        results = {
            "test average reward": average_reward,
        }

        # make the video
        if render:
            policy_plots = self.policy_model.get_monitor_data(exclude=self.policy_model.numeric_monitors)
            transition_plots = self.transition_model.get_monitor_data(exclude=self.transition_model.numeric_monitors)
            episode_videos = {"test episodes" : render_video(completed_framestacks)}
            prediction_videos = {"test prediction animations": animate_predictions(
                completed_episodes,
                self.transition_model,
                labels = env.get_attr('state_labels')[0],
                warmup=self.transition_config.learning.params.get('warmup_steps', 0),
                unroll=self.plotting_config.get("prediction_animation_unroll", 1),
                step=self.plotting_config.get("prediction_animation_step", 2),
            )}
            results.update(episode_videos)
            results.update(prediction_videos)
            results.update(policy_plots)
            results.update(transition_plots)

        prediction_results = baseline_prediction(
            transition_model=self.transition_model,
            episodes=completed_episodes,
            warmup=self.transition_config.learning.params.get('warmup_steps', 0),
            unroll=self.run_config.get("prediction_unroll", 1),
        )
        results.update(prediction_results)
            
        self.log(results, step=self.epochs)

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