import torch
from itertools import permutations
from random import shuffle

# 为了便于训练，以下函数都包括 batch 维度

def pfspStep(J:torch.tensor, state:torch.tensor) -> torch.tensor:
    """ 逐步递推形式获得指定解下的最大完工时间

    Args:
        J (torch.tensor): [batch_size, m_max] the next job
        state (torch.tensor): [batch_size, m_max] current state

    Returns:
        torch.tensor: the next state
    """

    m_max = J.shape[-1]
    
    next_state = torch.zeros_like(state)
    # [batch_size, m_max]
    
    next_state[:, 0] = state[:, 0] + J[:, 0]
    for i in range(1, m_max):
        next_state[:, i] = torch.maximum(state[:, i], next_state[:, i - 1]) + J[:, i]
    return next_state

def getCmax(pi:list, P:torch.tensor, state:torch.tensor=None) -> torch.tensor:
    """ 获得指定解下(pi)的最大完工时间

    Args:
        pi (list): [n] the processing sequence
        P (torch.tensor): [batch_size, n, m_max] the processing time matrix
        state (torch.tensor, optional): [batch_size, m_max] initial state. Defaults to [0, 0, ... ,0].

    Returns:
        torch.tensor: [batch_size] the maximum completion time (Cmax)
    """
    batch_size = P.shape[0]
    m_max = P.shape[-1]

    if state is None:
        state = torch.zeros(batch_size, m_max)
    for idx in pi:
        next_state = pfspStep(P[:, idx], state)
        state = next_state
    return state[:, - 1]

def getMinCmax(P:torch.tensor, state:torch.tensor=None) -> torch.tensor:
    """ 通过全部枚举的方法获得最小完工时间，适用于 n 较小的情况

    Args:
        P (torch.tensor): [batch_size, n, m_max] the processing time matrix
        state (torch.tensor, optional): [batch_size, m_max] initial state. Defaults to [0, 0, ... ,0].

    Returns:
        torch.tensor: [batch_size] the minimum Cmax
    """
    
    batch_size = P.shape[0]
    n = P.shape[1]

    res = torch.full((batch_size,), 1e9).type_as(P)
    for pi in permutations(range(n)):
        res = torch.minimum(res, getCmax(pi, P, state))
    return res

def getMeanCmax(P:torch.tensor, state:torch.tensor=None) -> torch.tensor:
    """ 通过随机采样获得 Epi(Cmax) 用于初始化 critictor

    Args:
        P (torch.tensor): [batch_size, n, m_max] the processing time matrix
        state (torch.tensor, optional): [batch_size, m_max] initial state. Defaults to [0, 0, ... ,0].

    Returns:
        torch.tensor: [batch_size] the expectation of Cmax
    """
    batch_size = P.shape[0]
    n = P.shape[1]

    res = torch.zeros(batch_size).type_as(P)
    for i in range(1000):
        pi = list(range(n))
        shuffle(pi)
        res += getCmax(pi, P, state)
    res /= 1000
    return res

def lowerBound(P:torch.tensor, state:torch.tensor=None) -> torch.tensor:
    """ 获得 PFSP 问题的一个简单下限

    Args:
        P (torch.tensor): [batch_size, n, m_max] the processing time matrix
        state (torch.tensor, optional): [batch_size, m_max] initial state. Defaults to [0, 0, ... ,0].

    Returns:
        torch.tensor: [batch_size] the lower bound of Cmax
    """
    batch_size = len(P)
    m_max = P.shape[-1]
    
    if state is None:
        state = torch.zeros(batch_size, m_max)

    res = torch.zeros_like(P[:, 0, 0])
    for i in range(batch_size):
        p = P[i]
        idx = (p.sum(0) + state[i]).argmax()
        p_follow = p[:, idx + 1:]
        res[i] = (p.sum(0) + state[i]).max() + p_follow.sum(1).min() - 1e-5

    return res