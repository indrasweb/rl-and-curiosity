
# coding: utf-8

# In[1]:


import datetime
from time import time

import numpy as np

import torch as T
from torch.autograd import Variable
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines.common.running_mean_std import RunningMeanStd

from itertools import count
from sys import argv


# In[2]:


DEVICE = T.device('cuda' if T.cuda.is_available() else 'cpu')
ENV_NAME = argv[1].split('.')[1]
NENV = 8
SEED = 420
set_global_seeds(420)


# In[3]:


class Logger:

    def __init__(self, print_rate=250):
        self.log = {'ep_r':[], 'ep_l':[], 'loss':[], 'pgloss':[], 
                    'vloss':[], 'ent':[]}
        self.n_ep = 0              # total games/episodes
        self.n_update = 1          # total weight updates
        self.n_frames = 0          # env steps (total from checkpoint)
        self.run_frames = 0        # env steps (for this run)
        self.max_rwd = -np.inf     # max rwd out of all games played
        self.start_time = time()   # time we started *this* run
        self.last_checkpoint = 0   # total_frames at last checkpoint
        self.print_rate = print_rate

    def eta(self):  # get hh:mm:ss left to train
        elapsed_time = time() - self.start_time
        frames_left = TOTAL_FRAMES - self.n_frames
        sec_per_frame = elapsed_time / self.n_frames
        sec_left = int(frames_left * sec_per_frame)
        eta_str = str(datetime.timedelta(seconds=sec_left))
        return eta_str

    def fps(self):  # get frames per second
        elapsed_time = time() - self.start_time
        fps = int(self.run_frames / elapsed_time)
        return fps

    def sma(self, x):  # simple moving average
        div = 200 if len(x) > 200 else len(x)
        return sum(list(zip(*x[-div:]))[-1])/div

    def print_log(self):
        fps = self.fps()
        eta = self.eta()
        print('-'*10, self.n_update, '/', TOTAL_UPDATES, '-'*10)
        print('Num Games:', self.n_ep)
        print('Num Frames:', self.n_frames)
        print('FPS:', fps)
        print('ETA:', eta)
        print('SMA Length:', self.sma(self.log['ep_l']))
        print('SMA Reward:', self.sma(self.log['ep_r']))
        print('SMA Entropy:', self.sma(self.log['ent']))
        print('SMA Loss:', self.sma(self.log['loss']))
        print('SMA PG Loss:', self.sma(self.log['pgloss']))
        print('SMA V Loss:', self.sma(self.log['vloss']))
        print('Max reward:', self.max_rwd)

    def record(self, ep, loss, pgloss, vloss, ent):
        
        self.n_update += 1
        self.n_frames += STEPS_PER_ROLLOUT
        self.run_frames += STEPS_PER_ROLLOUT
        fr = (self.n_frames, self.n_update)

        # stats about finished episodes/games
        for l, r in zip(ep['l'], ep['r']):
            self.log['ep_l'].append(fr+(l,))
            self.log['ep_r'].append(fr+(r,))
            if r > self.max_rwd: self.max_rwd = r
            self.n_ep += 1
             
        # nn training statistics
        self.log['loss'].append(fr+(loss,))
        self.log['pgloss'].append(fr+(pgloss,))
        self.log['vloss'].append(fr+(vloss,))
        self.log['ent'].append(fr+(ent,))
        
        # print log
        if self.n_update % self.print_rate == 0:
            self.print_log()


# In[4]:


class AC(nn.Module):
  
    def __init__(self, input_shape, num_actions):
        super().__init__()
        h, w, c = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
        )
        
        f = self.conv_size(self.conv, (c,h,w))

        self.flat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f, 512),
            nn.ReLU(True)
        )

        self.backbone = nn.Sequential(self.conv, self.flat)
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def conv_size(self, net, in_shape):
        x = Variable(T.rand(1, *in_shape))
        o = net(x)
        b = (-1, o.size(1), o.size(2), o.size(3))
        return o.data.view(1, -1).size(1)

    def forward(self, x):
        latent = self.backbone(x)
        return self.actor(latent), self.critic(latent)


# In[5]:


def ob_to_torch(x):
    x = np.moveaxis(x, -1, 1)
    x = T.from_numpy(x).float()
    x = x.to(DEVICE)
    return x


# In[6]:


def load_checkpoint(file_name):
    checkpoint = T.load(file_name, map_location=T.device('cpu'))
    
    venv = make_atari_env(ENV_NAME, num_env=NENV, seed=SEED)
    venv = VecFrameStack(venv, n_stack=4)
    env = checkpoint['env']
    env.set_venv(venv)
    in_dim = env.observation_space.shape
    policy_dim = env.action_space.n

    ac = AC(in_dim, policy_dim).to(DEVICE)
    ac.load_state_dict(checkpoint['ac'])
    ac_optimizer = Adam(ac.parameters(), 7e-4, eps=1e-5)
    ac_optimizer.load_state_dict(checkpoint['ac_opt'])

    logger = checkpoint['logger']

    return env, ac, ac_optimizer, logger


# In[7]:


# env, ac, ac_optimizer, logger = new_run()
env, ac, ac_optimizer, logger = load_checkpoint(argv[1])


# In[8]:


ob = env.reset()
for _ in range(5000):
    env.venv.venv.envs[0].render()
    ob = ob_to_torch(ob)
    logits, _ = ac(ob)
    dist = Categorical(logits=logits)
    action = dist.sample()
    ob, _,_,_ = env.step(action)

