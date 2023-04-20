from operator import itemgetter
from typing import Any, Dict, Optional, Union

import torch

from offlinerllib.module.net.attention.dt import DecisionTransformer
from offlinerllib.policy import BasePolicy
from offlinerllib.utils.misc import convert_to_tensor
from decision_rwkv.module.net.attention.rwkv import DecisionRWKV

class DecisionRWKVPolicy(BasePolicy):
    """
    DecisionRWKV, by substituting the GPT2 with RWKV Architecture.  
    """
    def __init__(
        self, 
        drwkv: DecisionRWKV, 
        dt_optim: torch.optim, 
        state_dim: int, 
        action_dim: int, 
        seq_len: int, 
        episode_len: int, 
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.drwkv = drwkv
        self.dt_optim = dt_optim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.episode_len = episode_len

        # we maintain a hidden and cell state for RNN mode testing
        self.hidden = None
        self.cell_state = None
        self.last_action = torch.zeros([1, 1, action_dim]).to(device)
        self.timestep = 0
        
        self.to(device)
        
    def configure_optimizers(self, lr, warmup_steps):
        self.drwkv_optim = torch.optim.AdamW(self.drwkv.parameters(), lr=lr)
        self.drwkv_optim_scheduler = torch.optim.lr_scheduler.LambdaLR(self.drwkv_optim, lambda step: min((step+1)/warmup_steps, 1))

    @torch.no_grad()
    def select_action(self, states, returns_to_go):
        states = torch.from_numpy(states).float().reshape(1, 1, self.state_dim).to(self.device)
        returns_to_go = torch.from_numpy(returns_to_go).float().reshape(1, 1, 1).to(self.device)
        timesteps = torch.tensor(self.timestep).long().reshape(1, 1, 1).to(self.device)
        
        action_pred, new_hidden, new_cell_state = self.drwkv(
            states=states, 
            actions=self.last_action, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            hiddens=self.hidden, 
            cell_state=self.cell_state
        )
        
        self.last_action = action_pred
        self.hidden = new_hidden
        self.cell_state = new_cell_state
        self.timestep += 1
        
        return action_pred[0, 0].squeeze().cpu().numpy()
    
    def reset(self):
        self.hidden = None
        self.cell_state = None
    
    def update(self, batch: Dict[str, Any], clip_grad: Optional[float]=None):
        for _key, _value in batch.items():
            batch[_key] = convert_to_tensor(_value, self.device)
        obss, actions, returns_to_go, timesteps, hiddens, cell_states, masks = \
            itemgetter("observations", "last_actions", "returns", "timesteps", "hiddens", "cell_states", "masks")(batch)
        
        action_pred, new_hidden, new_cell_state = self.drwkv(
            states=obss, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            hiddens=hiddens, 
            cell_states=cell_states
        )
        mse_loss = torch.nn.functional.mse_loss(action_pred, actions.detach(), reduction="none")
        mse_loss = (mse_loss * masks.unsqueeze(-1)).mean()
        self.drwkv_optim.zero_grad()
        mse_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.dt.parameters(), clip_grad)
        self.drwkv_optim.step()
        self.drwkv_optim_scheduler.step()
        
        return {
            "loss/mse_loss": mse_loss.item(), 
            "misc/learning_rate": self.drwkv_optim_scheduler.get_last_lr()[0]
        }, new_hidden, new_cell_state