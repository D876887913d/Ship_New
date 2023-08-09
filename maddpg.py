import torch
import copy

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

GAMMA=0.9

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3_mean = nn.Linear(64, action_size)
        self.fc3_std = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3_mean(x))  # Assume the action space is [-1, 1]
        std = F.softplus(self.fc3_std(x))  # Ensure standard deviation is positive
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        return action

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q_value = self.fc3(x)
        return Q_value

class MADDPG:
    def __init__(self, num_agents, state_size, action_size):
        self.batch_size=1024

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size

        # Initialize actor and critic networks for each agent
        self.actors = [Actor(state_size, action_size) for _ in range(num_agents)]
        self.critics = [Critic(num_agents * state_size, num_agents * action_size) for _ in range(num_agents)]

        # Initialize target networks
        self.target_actors = [copy.deepcopy(actor) for actor in self.actors]
        self.target_critics = [copy.deepcopy(critic) for critic in self.critics]

        # Initialize optimizers
        self.actor_optimizers = [torch.optim.Adam(actor.parameters()) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters()) for critic in self.critics]

    def update(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Convert to tensors     
        states = torch.tensor(states, dtype=torch.float32)  
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)  # Add extra dimension
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)  # Add extra dimension
        actions = torch.tensor(actions, dtype=torch.float32)

        # Update each agent
        for i in range(self.num_agents):
            # Compute target actions
            target_actions = [self.target_actors[j].sample(next_states[:, j*4:j*4+4]) for j in range(self.num_agents)]
            target_actions = torch.cat(target_actions, dim=-1)

            # Compute target Q-values
            target_Q = self.target_critics[i](next_states.view(self.batch_size, -1), target_actions)
            target_Q = rewards[:, i:i+1,0] + (1 - dones[:, i:i+1,0]) * GAMMA * target_Q

            # Compute current Q-values
            current_Q = self.critics[i](states.view(self.batch_size, -1), actions.view(self.batch_size, -1))

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q.detach())

            # Update critic
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()

            # Compute actor loss
            predicted_actions = [self.actors[j].sample(states[:, j*4:j*4+4]) if j == i 
                                else self.actors[j].sample(states[:,j*4:j*4+4]).detach() 
                                for j in range(self.num_agents)]
            predicted_actions = torch.cat(predicted_actions, dim=-1)
            actor_loss = -self.critics[i](states.view(self.batch_size, -1), predicted_actions).mean()

            # Update actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

            # Update target networks
            self.soft_update(self.target_actors[i], self.actors[i])
            self.soft_update(self.target_critics[i], self.critics[i])

    def soft_update(self, target, source, tau=0.01):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def act(self, states):
        states = torch.tensor(states, dtype=torch.float32)
        actions = [self.actors[i].sample(states[i*4:i*4+4]).detach().numpy() for i in range(self.num_agents)]
        return actions
    
