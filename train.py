from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
from ddpg_agent import Agent

env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)


def ddpg(n_episodes=500, max_t=1000, print_every=100, target_score=30):

    scores_deque = deque(maxlen=print_every)
    scores_plot = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones, t)
            states = next_states
            score += rewards
            if np.any(dones):
                break

        scores_deque.append(np.mean(score))
        scores_plot.append(np.mean(score))

        print('\rEpisode {}\tMean: {:.2f}\tMin: {:.2f}\tMax: {:.2f}\tRunning Average: {:.2f}'.format(i_episode, np.mean(score), np.min(score), np.max(score), np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tMean: {:.2f}\tMin: {:.2f}\tMax: {:.2f}'.format(i_episode, np.mean(scores_deque), np.min(scores_deque), np.max(scores_deque)))

        if i_episode > 100 and np.mean(scores_deque) >= target_score:
            print('\nEnvironment solved in {} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break

    return scores_plot


scores = ddpg()

env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
