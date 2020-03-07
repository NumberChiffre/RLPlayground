import torch
from collections import defaultdict
from torch.distributions import Categorical
import numpy as np
from gym import Env


def nested_d():
    """for any arbitrary number of levels"""
    return defaultdict(nested_d)


# Thanks Vlad!
def torch_argmax_mask(q: torch.Tensor, dim: int):
    """ Returns a random tie-breaking argmax mask
    Example:
        >>> import torch
        >>> torch.manual_seed(1337)
        >>> q = torch.ones(3, 2)
        >>> torch_argmax_mask(q, 1)
        # tensor([[False,  True],
        #         [ True, False],
        #         [ True, False]])
        >>> torch_argmax_mask(q, 1)
        # tensor([[False,  True],
        #         [False,  True],
        #         [ True, False]])
    """
    rand = torch.rand_like(q)
    if dim == 0:
        mask = rand * (q == q.max(dim)[0])
        mask = mask == mask.max(dim)[0]
        assert int(mask.sum()) == len(q.shape)
    elif dim == 1:
        mask = rand * (q == q.max(dim)[0].unsqueeze(1).expand(q.shape))
        mask = mask == mask.max(dim)[0].unsqueeze(1).expand(q.shape)
        assert int(mask.sum()) == int(q.shape[0])
    else:
        raise NotImplemented("Only vectors and matrices are supported")
    return mask


def get_epsilon_dist(eps: float, env: Env, observation: torch.Tensor,
                     model: torch.nn.Module) -> Categorical:
    """get the probability distributions of the q-value"""
    q = model(observation)
    probs = torch.empty_like(q).fill_(
        eps / (env.action_space.n - 1))
    probs[torch_argmax_mask(q, len(q.shape) - 1)] = 1 - eps
    return Categorical(probs=probs)


def get_epsilon(eps_start: float, eps_final: float, eps_decay: float,
                t: int) -> float:
    """use decay for epsilon exploration"""
    return eps_final + (eps_start - eps_final) * np.exp(-1.0 * t / eps_decay)


def soft_update(value_net: torch.nn.Module, target_net: torch.nn.Module,
                tau: float):
    """update each training step by a hyperparameter adjustment"""
    for t_param, v_param in zip(target_net.parameters(),
                                value_net.parameters()):
        if t_param is v_param:
            continue
        new_param = tau * v_param.data + (1.0 - tau) * t_param.data
        t_param.data.copy_(new_param)


def hard_update(value_net: torch.nn.Module, target_net: torch.nn.Module):
    """update each training step by a full update, based on an update frequency"""
    for t_param, v_param in zip(target_net.parameters(),
                                value_net.parameters()):
        if t_param is v_param:
            continue
        new_param = v_param.data
        t_param.data.copy_(new_param)


