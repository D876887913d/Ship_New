import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer.

        Args:
        - buffer_size (int): Maximum number of experiences to store in the buffer. If the buffer is full, old experiences are removed.
        - batch_size (int): Number of experiences to sample in each batch.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=int(buffer_size))

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Args:
        - state (np.array): The state.
        - action (np.array): The action.
        - reward (float): The reward.
        - next_state (np.array): The next state.
        - done (bool): Whether the episode is done.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Returns:
        - states (np.array): Batch of states.
        - actions (np.array): Batch of actions.
        - rewards (np.array): Batch of rewards.
        - next_states (np.array): Batch of next states.
        - dones (np.array): Batch of done flags.
        """
        experiences = random.sample(self.buffer, self.batch_size)

        states = np.array([exp[0] for exp in experiences])
        actions = np.array([exp[1] for exp in experiences])
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)