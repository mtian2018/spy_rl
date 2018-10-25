import logging
from itertools import count

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from matplotlib import pyplot as plt

import stock_gym as gym


class PolGrad:
    """real training here"""

    def __init__(self):
        self.optimizer = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE)
        self.state_tape = []
        self.action_tape = []
        self.reward_tape = []

    def record(self, state, action, reward):
        self.state_tape.append(state)
        self.action_tape.append(action)
        self.reward_tape.append(reward)

    def reset(self):
        self.state_tape.clear()
        self.action_tape.clear()
        self.reward_tape.clear()

    def learn(self):
        self._cal_reward()

        self.optimizer.zero_grad()
        for i in range(len(self.state_tape)):
            state = self.state_tape[i]
            action = self.action_tape[i] # use original action
            reward = self.reward_tape[i]

            prob = net(state)
            dist = Categorical(prob)
            loss = -dist.log_prob(action) * reward
            loss.backward()

        self.optimizer.step()
        self.reset()

    def _cal_reward(self):
        """accumulate and adjust scores"""

        serial = range(len(self.reward_tape))
        last_item = 0
        for i in reversed(serial):
            if self.reward_tape[i] == 0:
                last_item = 0
            else:
                self.reward_tape[i] += last_item * GAMMA
                last_item = self.reward_tape[i]

        reward_mean = np.mean(self.reward_tape)
        reward_std = np.std(self.reward_tape)

        for i in serial:
            self.reward_tape[i] = (self.reward_tape[i] - reward_mean) / reward_std
        # self.reward_tape = (self.reward_tape - reward_mean) / reward_std


def train():
    """running environment"""

    pol_grad = PolGrad()
    batch_duration = []

    act_name = {0: 'sell', 1: 'hold', 2: 'buy'}
    for episode in range(200):

        next_state = env.reset()    # use next_state to be clear

        for step in count():
            state = torch.from_numpy(next_state).float()

            prob = net(state)
            dist = Categorical(prob)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item() - 1)
            pol_grad.record(state, action, 0 if done else reward)
            # print(f"action: {act_name[action.item()]}\treward: {reward:.2f}")


            if done:
                batch_duration.append(reward)
                break

        logger.info(f'Episode {episode} action {act_name[action.item()]} reward: {reward:.2f}')
        if (episode+1) % BATCH_SIZE == 0:
            pol_grad.learn()

    plt.plot(batch_duration, '.')
    plt.title('categorical')
    plt.show()

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    env = gym.Quotes()

    # hyper parameters
    LEARNING_RATE = 0.01
    GAMMA = 0.99
    N_IN, N_OUT = env.enf_pram()
    BATCH_SIZE = 5

    # net = nn.Sequential(
    #     nn.Linear(N_IN, N_IN * 6),
    #     nn.ReLU(),
    #     nn.Linear(N_IN * 6, N_IN * 6),
    #     nn.ReLU(),
    #     nn.Linear(N_IN * 6, N_OUT),
    #     nn.Sigmoid(),
    #     # nn.Softmax(),
    # )
    net = nn.Sequential(
        nn.Linear(N_IN, 128, bias=False),
        nn.Dropout(p=0.1),
        nn.ReLU(),
        nn.Linear(128, N_OUT),
        nn.Softmax(),
    )

    train()
