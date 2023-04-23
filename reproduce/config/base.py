from UtilsRL.misc import NameSpace

seed = 42

betas = [0.9, 0.999]
clip_grad = 0.25
episode_len = 1000

normalize_obs = True
normalize_reward = False

embed_dim = 128
seq_len = 20
num_layers = 3

max_epoch = 1000
step_per_epoch = 100
eval_episode = 10
eval_interval = 10
relabel_interval = 1000000
log_interval = 10
save_interval = 50
warmup_steps = 10000

name = "original"
class wandb(NameSpace):
    entity = None
    project = None

debug = False

batch_size = 64
lr = 1e-4
return_scale = 1000.0

mode = "GPT" # choice from {"RNN", "GPT"}