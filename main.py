from shipenv import *
from maddpg import *
from replay_buffer import ReplayBuffer  # Assuming you have a ReplayBuffer class defined in replay_buffer.py

if __name__=="__main__":
    env= Environment()

    # Initialize MADDPG
    maddpg = MADDPG(num_agents=env.num_agents, state_size=env.observation_space.shape[0]//env.num_agents, action_size=env.action_space.shape[0]//env.num_agents)

    # Initialize ReplayBuffer
    buffer = ReplayBuffer(buffer_size=1e6, batch_size=1024)

    # Number of episodes
    num_episodes = 1000
    
    # Training loop
    for i_episode in range(num_episodes):
        # Reset the environment
        states = env.reset()

        # Reset the score
        scores = np.zeros(2)

        i=0

        while i<488:
            # Select actions
            actions = maddpg.act(states)

            # Take a step in the environment
            next_states, rewards, dones = env.step(actions)
            
            # Add experience to ReplayBuffer
            buffer.add(states, actions, rewards, next_states, dones)
            
            # Update the score
            scores += rewards

            # Update the state
            states = next_states

            # Check if the episode is done
            if np.any(dones):
                break
            # print()
            i+=1

        # Update MADDPG
        if len(buffer) >= buffer.batch_size:
            experiences = buffer.sample()
            maddpg.update(experiences)

        

        # Print the score
        print('Episode {}: {}'.format(i_episode, scores))