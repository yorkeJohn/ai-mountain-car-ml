'''
Author: John Yorke
CSCI 3482: Artificial Intelligence
Saint Mary's University
'''

import gymnasium as gym
import numpy as np
import time

class QLAgent:
    '''
    A class implementing Q-learning for Gymnasium Mountain Car
    '''
    def __init__(self, env, num_states, alpha, gamma, Q=None) -> None:
        '''
        Params:
            env (Env): The gym enviornment
            num_states (int): The number of discrete states
            alpha (float): The learning rate 
            gamma (float): The discount factor
            Q (dict): Initial Q-values
        '''
        self.env = env
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        # use given Q-values or initialize as a random value in the range [-2, 0]
        self.Q = Q if Q is not None else np.random.uniform(low=-2, high=0, size=(num_states, num_states, env.action_space.n))
        # state ranges
        min_p, min_v = self.env.observation_space.low
        max_p, max_v = self.env.observation_space.high
        # subtract a small amount to account for the left bin edge
        self.p_bins = np.linspace(start=min_p-0.1, stop=max_p, num=self.num_states)
        self.v_bins = np.linspace(start=min_v-0.1, stop=max_v, num=self.num_states)

    def get_d_state(self, state) -> tuple:
        '''
        Gets a discrete state for a given non-discrete state

        Params: 
            state (tuple): The given state
        Returns:
            tuple: A discrete state of the form (position, velocity)
        '''
        p, v = state
        return (np.digitize(p, self.p_bins, right=True)-1, np.digitize(v, self.v_bins, right=True)-1)

    def choose_action(self, state) -> int:
        '''
        Chooses the action with the max Q-value for a given state

        Params:
            state (tuple): The given state
        Returns:
            int: The best action
        '''
        p, v = self.get_d_state(state)
        return np.argmax(self.Q[p, v])

    def update_Q(self, state, action, reward, next_state) -> None:
        '''
        Executes a Q-update for a given state

        Params:
            state (tuple): The given state
            action (int): The action taken
            reward (float): The reward received
            next_state (tuple): The next expected state
        Returns:
            None
        '''
        p, v = self.get_d_state(state)
        next_p, next_v = self.get_d_state(next_state)
        max_Q = np.max(self.Q[next_p, next_v])
        sample = reward + self.gamma * max_Q
        self.Q[p, v][action] = (1 - self.alpha) * self.Q[p, v][action] + self.alpha * sample

    def run_episode(self) -> dict:
        '''
        Runs a Q-learning episode. The episode ends when the goal is reached.

        Returns:
            dict: The resulting Q-values
        '''
        state, *_ = self.env.reset()
        state = tuple(state)
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, *_ = self.env.step(action)
            self.update_Q(state, action, reward, tuple(next_state))
            state = next_state
        return self.Q

if __name__ == '__main__':
    # episode options
    num_episodes = 100
    display_interval = 10
    # env options
    num_states = 10 # discrete state space size
    alpha = 0.1 # learning rate
    gamma = 0.99 # discount

    # execute episodes
    Q = None
    for episode in range(1, num_episodes + 1):
        # only render if display interval episode
        display = episode % display_interval == 0
        env = gym.make('MountainCar-v0', render_mode='human' if display else None)
        agent = QLAgent(env, num_states, alpha, gamma, Q)
        
        # run the episode
        print(f'Running episode {episode}...')
        start_time = time.time()
        Q = agent.run_episode()

        end_time = time.time()
        delta = end_time - start_time
        print(f'Goal reached in {delta:.4f} seconds')
        env.close()
    
    print('All episodes complete')
