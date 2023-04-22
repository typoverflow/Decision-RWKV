from typing import Callable, Dict, List

import gym
import numpy as np
import torch
import torch.nn as nn

@torch.no_grad()
def eval_decision_rwkv(
    env: gym.Env, actor: nn.Module, target_returns: List[float], return_scale: float, n_episode: int, seed: int, score_func=None
) -> Dict[str, float]:
    
    def eval_one_return(target_return, score_func=None):
        if score_func is None:
            score_func = env.get_normalized_score
        env.seed(seed)
        actor.eval()
        episode_lengths = []
        episode_returns = []
        for _ in range(n_episode):
            actor.reset()
            state, done = env.reset(), False
            return_to_go = target_return / return_scale
            
            episode_return = episode_length = 0
            for step in range(actor.episode_len):
                action = actor.select_action(
                    state, 
                    np.asarray(return_to_go)
                )
                next_state, reward, done, info = env.step(action)
                state = next_state
                return_to_go = return_to_go - reward/return_scale
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    episode_returns.append(score_func(episode_return)*100)
                    episode_lengths.append(episode_length)
                    break
                
        actor.train()
        episode_returns = np.asarray(episode_returns)
        episode_lengths = np.asarray(episode_lengths)
        return {
            "normalized_score_mean_target{:.1f}".format(target_return): episode_returns.mean(), 
            "normalized_score_std_target{:.1f}".format(target_return): episode_returns.std(), 
            "length_mean_target{:.1f}".format(target_return): episode_lengths.mean(), 
            "length_std_target{:.1f}".format(target_return): episode_lengths.std()
        }
    
    ret = {}
    for target in target_returns:
        ret.update(eval_one_return(target, score_func))
    return ret