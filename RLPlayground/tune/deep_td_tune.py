import gym
import numpy as np
import torch
from collections import deque
from typing import Dict
import os

from ray import tune

from RLPlayground.agents.agent import Agent


# TODO: decouple config into Agent/Experiment and define params to tune
class DeepTDTrainable(tune.Trainable):
    """Abstract class for trainable models, functions, etc.

    A call to ``train()`` on a trainable will execute one logical iteration of
    training. As a rule of thumb, the execution time of one train call should
    be large enough to avoid overheads (i.e. more than a few seconds), but
    short enough to report progress periodically (i.e. at most a few minutes).

    Calling ``save()`` should save the training state of a trainable to disk,
    and ``restore(path)`` should restore a trainable to the given state.

    Generally you only need to implement ``_train``, ``_save``, and
    ``_restore`` here when subclassing Trainable.

    Note that, if you don't require checkpoint/restore functionality, then
    instead of implementing this class you can also get away with supplying
    just a ``my_train(config, reporter)`` function to the config.
    The function will be automatically converted to this interface
    (sans checkpoint functionality).
    """

    def _setup(self, config: Dict):
        self.experiment_cfg, self.agent_cfg = config['experiment_cfg'], config[
            'agent_cfg']
        env = gym.make(config['env_name'])
        self.agent = Agent.build(type=config['agent'], env=env,
                                 params=self.agent_cfg)
        self.track_reward_100 = deque(
            maxlen=config['consecutive_steps_to_solve'])
        self.episodes = 0

    def _train(self):
        # use for episodic training
        self.episodes += 1
        generator_obj = self.agent.learn(
            num_steps=self.experiment_cfg['steps'])
        episode_result = next(generator_obj)
        self.track_reward_100.append(episode_result['cum_reward'])
        episode_result['track_reward_100'] = np.mean(self.track_reward_100)
        episode_result['below_reward_thresh'] = False
        if self.episodes > 1000:
            episode_result['below_reward_thresh'] = episode_result[
                                                        'track_reward_100'] < 50
        return episode_result

    def _save(self, checkpoint_dir: str) -> str:
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.agent.value_net.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path: str):
        self.model.load_state_dict(torch.load(checkpoint_path))
