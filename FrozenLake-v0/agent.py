import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make("FrozenLake-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n


episodes = 10000
max_steps_per_episode = 100

alpha = 0.1 #learning rate
gamma = 0.99 #discount rate

epsilon = 1 #exploration rate
max_epsilon = 1
min_epsilon = 0.01
exploration_decay_rate = 0.001

q_table = np.zeros((state_space_size, action_space_size))

rewards_all_episodes = []

# Q-learning algorithm
for episode in range(episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode): 

    # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > epsilon:
            action = np.argmax(q_table[state,:]) 
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        #Update Q using bellman equation
        q_table[state, action] = q_table[state, action] * (1 - alpha) + alpha * (reward + gamma * np.max(q_table[new_state, :]))
        # q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.nanargmax(q_table[new_state,:]))


        state = new_state
        rewards_current_episode += reward 

        if done: 
            break

    #Calculate exploration decay, update epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)    

rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),episodes/1000)
count = 1000
#Completed run, print out results
print("********Avr reward / thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000