'''
Author: gongweijing 876887913@qq.com
Date: 2023-12-08 22:28:45
LastEditors: gongweijing 876887913@qq.com
LastEditTime: 2023-12-08 22:38:10
FilePath: /Ship_New/maddpg/simple_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.torch.model import Model

class MAModel(Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 critic_dim,
                 continuous_actions=False
                 ):
        super(MAModel,self).__init__()
        self.actor=Actor(obs_dim,act_dim,continuous_actions)
        self.critic=Critic(critic_dim)
    
    def policy(self,obs):
        return self.actor(obs)

    def value(self,obs,act):
        return self.critic(obs,act)
    
    def get_actor_params(self):
        return self.actor.parameters()
    
    def get_critic_params(self):
        return self.critic.parameters()

# input:  agent_i_obs_dim
# output: agent_i_action_dim
class Actor(Model):
    def __init__(self, obs_dim,act_dim,continuous_actions=False):
        super(Actor, self).__init__()
        self.continuous_actions=continuous_actions
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, act_dim)
        if continuous_actions:
            self.std = nn.Linear(64, act_dim)

    def forward(self, x):
        hid1 = F.relu(self.fc1(x))
        hid2 = F.relu(self.fc2(hid1))
        means = self.fc3(hid2)
        if self.continuous_actions:
            std=self.std(hid2)
            return (means,std)
        return means

# input:  all_obs_dim+all_action_dim
# output: 1  (Q-value)
class Critic(Model):
    def __init__(self, critic_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(critic_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat(state+action, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        q_value = torch.squeeze(q_value,dim=1)
        return q_value
