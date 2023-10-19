import numpy as np
import gym

# Initialize the FrozenLake environment
env = gym.make('FrozenLake-v0')
state_size = env.observation_space.n
action_size = env.action_space.n

# Initialize Q-table with zeros
qtable = np.zeros((state_size, action_size))
learning_rate = 0.8
gamma = 0.95
num_episodes = 10000

for i in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(qtable[state, :] + np.random.randn(1, action_size) * (1.0 / (i+1)))
        new_state, reward, done, _ = env.step(action)
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        state = new_state

# Test the learned policy
num_test_episodes = 3
for i in range(num_test_episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(qtable[state, :])
        new_state, _, done, _ = env.step(action)
        env.render()
        state = new_state
