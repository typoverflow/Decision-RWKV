from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from offlinerllib.buffer.base import Buffer
from offlinerllib.utils.functional import discounted_cum_sum

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class D4RLTrajectoryBuffer(Buffer, IterableDataset):
    def __init__(
        self, 
        dataset, 
        seq_len: int, 
        num_layers: int, 
        embed_dim: int, 
        discount: float=1.0, 
        return_scale: float=1.0,
        device="cpu", 
    ) -> None:
        converted_dataset = {
            "observations": dataset["observations"].astype(np.float32), 
            "actions": dataset["actions"].astype(np.float32), 
            "rewards": dataset["rewards"][:, None].astype(np.float32), 
            "terminals": dataset["terminals"][:, None].astype(np.float32), 
            "next_observations": dataset["next_observations"].astype(np.float32)
        }
        traj, traj_len = [], []
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.discount = discount
        self.return_scale = return_scale
        self.device = device
        traj_start = 0
        for i in range(dataset["rewards"].shape[0]):
            if dataset["ends"][i]:
                episode_data = {k: v[traj_start:i+1] for k, v in converted_dataset.items()}
                episode_data["returns"] = discounted_cum_sum(episode_data["rewards"], discount=discount) * self.return_scale

                action_dim = episode_data["actions"].shape[-1]
                episode_data["last_actions"] = np.concatenate([np.zeros([1, action_dim]), episode_data["actions"][:-1]], axis=0)
                
                traj.append(episode_data)
                traj_len.append(i+1-traj_start)
                traj_start = i+1
        self.traj_len = np.array(traj_len)
        self.size = self.traj_len.sum()
        self.traj_num = len(self.traj_len)
        self.sample_prob = self.traj_len / self.size
        
        # pad trajs to have the same mask len
        self.max_len = self.traj_len.max() + self.seq_len - 1  # this is for the convenience of sampling
        for i_traj in range(self.traj_num):
            this_len = self.traj_len[i_traj]
            for _key, _value in traj[i_traj].items():
                traj[i_traj][_key] = pad_along_axis(_value, pad_to=self.max_len)
            traj[i_traj]["masks"] = np.hstack([np.ones(this_len), np.zeros(self.max_len-this_len)])
        
        # register all entries
        self.observations = np.asarray([t["observations"] for t in traj], dtype=np.float32)
        self.actions = np.asarray([t["actions"] for t in traj], dtype=np.float32)
        self.last_actions = np.asarray([t["last_actions"] for t in traj], dtype=np.float32)
        self.returns = np.asarray([t["returns"] for t in traj], dtype=np.float32)
        self.rewards = np.asarray([t["rewards"] for t in traj], dtype=np.float32)
        self.terminals = np.asarray([t["terminals"] for t in traj], dtype=np.float32)
        self.next_observations = np.asarray([t["next_observations"] for t in traj], dtype=np.float32)
        self.masks = np.asarray([t["masks"] for t in traj], dtype=np.float32)
        self.timesteps = np.arange(self.max_len)

        self.hiddens = np.zeros([self.traj_num, self.max_len, num_layers, embed_dim], dtype=np.float32)
        self.cell_states = np.stack([
            np.zeros([self.traj_num, self.max_len, num_layers, embed_dim]), 
            np.ones([self.traj_num, self.max_len, num_layers, embed_dim]) * (-1e38)
        ], axis=-1).reshape([self.traj_num, self.max_len, num_layers, 2*embed_dim]).astype(np.float32)

    def __len__(self):
        return self.size

    def __prepare_sample(self, traj_idx, start_idx):
        return {
            "observations": self.observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "actions": self.actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "last_actions": self.last_actions[traj_idx, start_idx:start_idx+self.seq_len], 
            "returns": self.returns[traj_idx, start_idx:start_idx+self.seq_len], 
            "rewards": self.rewards[traj_idx, start_idx:start_idx+self.seq_len], 
            "terminals": self.terminals[traj_idx, start_idx:start_idx+self.seq_len], 
            "next_observations": self.next_observations[traj_idx, start_idx:start_idx+self.seq_len], 
            "masks": self.masks[traj_idx, start_idx:start_idx+self.seq_len], 
            "hiddens": self.hiddens[traj_idx, start_idx:start_idx+self.seq_len], 
            "cell_states": self.cell_states[traj_idx, start_idx:start_idx+self.seq_len], 
            "timesteps": self.timesteps[start_idx:start_idx+self.seq_len], 
        }
    
    def __iter__(self):
        while True:
            traj_idx = np.random.choice(self.traj_num, p=self.sample_prob)
            start_idx = np.random.choice(self.traj_len[traj_idx])
            yield self.__prepare_sample(traj_idx, start_idx)
            
    def random_batch(self, batch_size: int):
        batch_data = {}
        traj_idxs = np.random.choice(self.traj_num, size=batch_size, p=self.sample_prob)
        start_idxs = []
        for i in range(batch_size):
            traj_idx = traj_idxs[i]
            start_idx = np.random.choice(self.traj_len[traj_idx])
            start_idxs.append(start_idx)
            sample = self.__prepare_sample(traj_idx, start_idx)
            for _key, _value in sample.items():
                if not _key in batch_data:
                    batch_data[_key] = []
                batch_data[_key].append(_value)
        for _key, _value in batch_data.items():
            batch_data[_key] = np.stack(_value, axis=0)
        return batch_data, traj_idxs, np.asarray(start_idxs)
    
    @torch.no_grad()
    def relabel(self, policy):
        policy.eval()
        non_mask_idx = np.ones([self.traj_num, ], dtype=np.bool8)
        timestep = 0
        cur_num = self.traj_num
        while True:
            non_mask_idx &= self.masks[:, timestep].astype(np.bool8)
            if not non_mask_idx.any():
                break
            cur_num = non_mask_idx.sum()
            cur_last_actions = torch.from_numpy(self.last_actions[non_mask_idx, timestep:timestep+1]).to(self.device)
            cur_states = torch.from_numpy(self.observations[non_mask_idx, timestep:timestep+1]).to(self.device)
            cur_returns_to_go = torch.from_numpy(self.returns[non_mask_idx, timestep:timestep+1]).to(self.device)
            cur_timesteps = (torch.ones([cur_num, 1], dtype=torch.long)*timestep).to(self.device)
            cur_hiddens = torch.from_numpy(self.hiddens[non_mask_idx, timestep]).to(self.device)
            cur_cell_states = torch.from_numpy(self.cell_states[non_mask_idx, timestep]).to(self.device)
            
            _, new_hiddens, new_cell_states = policy(
                last_actions=cur_last_actions, 
                states=cur_states, 
                returns_to_go=cur_returns_to_go, 
                timesteps=cur_timesteps, 
                hiddens=cur_hiddens, 
                cell_states=cur_cell_states
            )
            self.hiddens[non_mask_idx, timestep+1] = new_hiddens.cpu().numpy()
            self.cell_states[non_mask_idx, timestep+1] = new_cell_states.cpu().numpy()

            timestep += 1
        policy.train()