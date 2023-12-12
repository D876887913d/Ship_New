import torch
import numpy as np
from maddpg.replay import ReplayMemory
from maddpg.core.torch.agent import Agent

class MAAgent(Agent):
    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None):
        self.agent_index = agent_index
        self.obs_dim_n = obs_dim_n
        self.act_dim_n = act_dim_n
        self.batch_size = batch_size
        self.n = len(act_dim_n)

        self.memory_size = int(1e5)
        self.min_memory_size = batch_size * 25  # batch_size * args.max_episode_len
        self.rpm = ReplayMemory(
            max_size=self.memory_size,
            obs_dim=self.obs_dim_n[agent_index],
            act_dim=self.act_dim_n[agent_index])
        self.global_train_step = 0

        super(MAAgent, self).__init__(algorithm)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # Attention: In the beginning, sync target model totally.
        self.alg.sync_target(decay=0)

    def predict(self, obs):
        """ predict action by model
        """
        obs = torch.from_numpy(obs.reshape(1, -1)).float()
        act = self.alg.predict(obs)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def sample(self, obs, use_target_model=False):
        """ sample action by model or target_model
        """
        obs = torch.from_numpy(obs.reshape(1, -1)).float()
        act = self.alg.sample(obs, use_target_model=use_target_model)
        act_numpy = act.detach().cpu().numpy().flatten()
        return act_numpy

    def learn(self, agents):
        """ sample batch, compute q_target and train
        """
        self.global_train_step += 1

        # only update parameter every 100 steps
        if self.global_train_step % 100 != 0:
            return 0.0

        if self.rpm.size() <= self.min_memory_size:
            return 0.0

        batch_obs_n = []
        batch_act_n = []
        batch_obs_next_n = []

        # sample batch
        rpm_sample_index = self.rpm.make_index(self.batch_size)
        for i in range(self.n):
            batch_obs, batch_act, _, batch_obs_next, _ \
                = agents[i].rpm.sample_batch_by_index(rpm_sample_index)
            batch_obs_n.append(batch_obs)
            batch_act_n.append(batch_act)
            batch_obs_next_n.append(batch_obs_next)
        _, _, batch_rew, _, batch_isOver = self.rpm.sample_batch_by_index(rpm_sample_index)

        batch_obs_n = [
            torch.from_numpy(obs).float() for obs in batch_obs_n
        ]
        batch_act_n = [
            torch.from_numpy(act).float() for act in batch_act_n
        ]
        batch_rew = torch.from_numpy(batch_rew).float()
        batch_isOver = torch.from_numpy(batch_isOver).float()

        # compute target q
        target_act_next_n = []
        batch_obs_next_n = [
            torch.from_numpy(obs).float() for obs in batch_obs_next_n
        ]

        for i in range(self.n):
            target_act_next = agents[i].alg.sample(batch_obs_next_n[i], use_target_model=True)
            target_act_next = target_act_next.detach()
            target_act_next_n.append(target_act_next)


        target_q_next = self.alg.Q(batch_obs_next_n, target_act_next_n, use_target_model=True)
        target_q = batch_rew + self.alg.gamma * (1.0 - batch_isOver) * target_q_next.detach()
       
        # learn
        critic_cost = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        critic_cost = float(critic_cost.cpu().detach())

        return critic_cost

    def add_experience(self, obs, act, reward, next_obs, terminal):
        self.rpm.append(obs, act, reward, next_obs, terminal)
