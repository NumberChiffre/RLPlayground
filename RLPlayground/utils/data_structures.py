import numpy as np
from enum import Enum
from typing import NamedTuple, Union


class RLAlgorithm(Enum):
    VI = 'value_iteration'
    PI = 'policy_iteration'
    QLearning = 'q_learning'
    MONTE_CARLO = 'monte_carlo'
    EXPECTED_SARSA = 'expected_sarsa'
    SARSA = 'sarsa'


class TargetUpdate(Enum):
    HARD = 'hard'
    SOFT = 'soft'


class Transition(NamedTuple):
    # store state, action, reward, next_state, done as Transition tuple
    s0: np.ndarray
    a: Union[int, str]  # Action
    r: np.ndarray
    s1: np.ndarray
    done: bool = False
