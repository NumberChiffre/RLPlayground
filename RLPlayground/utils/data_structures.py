import numpy as np
from enum import Enum
from typing import NamedTuple, Union
from collections import defaultdict


class RLAlgorithm(Enum):
    VI = 'value_iteration'
    PI = 'policy_iteration'
    QLearning = 'q_learning'
    MONTE_CARLO = 'monte_carlo'
    EXPECTED_SARSA = 'expected_sarsa'
    SARSA = 'sarsa'


class Transition(NamedTuple):
    # store state, action, reward, next_state, done as Transition tuple
    s0: np.ndarray
    a: Union[int, str]  # Action
    r: np.ndarray
    s1: np.ndarray
    done: bool = False


# gotta put more keys into defaultdict + able to pickle..
def foo1():
    return defaultdict(dict)


def foo2():
    return defaultdict(int)


def foo():
    return defaultdict(foo1)