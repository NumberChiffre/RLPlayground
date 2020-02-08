from enum import Enum
from typing import NamedTuple, Union
import numpy as np


class RLAlgorithm(Enum):
    VI = 'value_iteration'
    PI = 'policy_iteration'
    QLearn = 'q_learning'
    MONTE_CARLO = 'monte_carlo'


class Experience(NamedTuple):
    # store state, action, reward, next_state, done as Experience tuple
    s0: np.ndarray
    a: Union[int, str]  # Action
    r: np.ndarray
    s1: np.ndarray
    done: bool = False
