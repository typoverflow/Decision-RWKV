import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import trange
from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger

from offlinerllib.utils.d4rl import get_d4rl_dataset

from decision_rwkv.buffer.rwkv_buffer import D4RLTrajectoryBuffer
from decision_rwkv.module.attention.rwkv import DecisionRWKV
from decision_rwkv.policy.model_free.DecisionRWKV import DecisionRWKVPolicy
from decision_rwkv.utils.eval import eval_decision_rwkv_rnn, eval_decision_rwkv_gpt

args = parse_args()
exp_name = "_".join([args.task, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path=f"./log/dt/offline/{args.name}", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward, discard_last=False)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

offline_buffer = D4RLTrajectoryBuffer(dataset, seq_len=args.seq_len, num_layers=args.num_layers, embed_dim=args.embed_dim, return_scale=args.return_scale, device=args.device)

drwkv = DecisionRWKV(
    obs_dim=obs_shape, 
    action_dim=action_shape, 
    embed_dim=args.embed_dim, 
    num_layers=args.num_layers, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
).to(args.device)

policy = DecisionRWKVPolicy(
    drwkv=drwkv, 
    state_dim=obs_shape, 
    action_dim=action_shape, 
    seq_len=args.seq_len, 
    episode_len=args.episode_len, 
    device=args.device
).to(args.device)
policy.configure_optimizers(args.lr, args.warmup_steps)


# main loop
policy.train()
for i_epoch in trange(1, args.max_epoch+1):
    for i_step in range(args.step_per_epoch):
        batch, traj_idxs, start_idxs = offline_buffer.random_batch(args.batch_size)
        train_metrics, new_hiddens, new_cell_states = policy.update(batch, clip_grad=args.clip_grad)

    if args.mode == "RNN" and i_epoch % args.relabel_interval == 0:
        offline_buffer.relabel(policy)

    if i_epoch % args.eval_interval == 0:
        if args.mode == "RNN":
            eval_metrics = eval_decision_rwkv_rnn(env, policy, args.target_returns, args.return_scale, args.eval_episode, seed=args.seed)
        else:
            eval_metrics = eval_decision_rwkv_gpt(env, policy, args.target_returns, args.return_scale, args.eval_episode, seed=args.seed)
        logger.info(f"Episode {i_epoch}: \n{eval_metrics}")
    
    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)
        
    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/dt/offline/{args.name}/{args.task}/seed{args.seed}/policy/")
    