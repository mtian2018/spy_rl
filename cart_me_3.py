import logging
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
import numpy as np
import gym
from itertools import count
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# hyper parameters
LEARNING_RATE = 0.01
GAMMA = 0.99

env = gym.make('CartPole-v1')
N_IN = env.observation_space.shape[0]

net = nn.Sequential(
    nn.Linear(N_IN, 24),
    nn.ReLU(),
    nn.Linear(24, 36),
    nn.ReLU(),
    nn.Linear(36, 1),
    nn.Sigmoid(),
)

# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.kaiming_normal_(m.weight)
#         if m.bias is not None:
#           m.bias.data.fill_(0.01)
#
# net.apply(init_weights)

class Recorder:
    """store running history"""

    def __init__(self):
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


class PolGrad:
    """real training here"""

    def __init__(self):
        self.optimizer = torch.optim.RMSprop(net.parameters(), lr=LEARNING_RATE)

    def learn(self, recorder):
        self._cal_reward(recorder.reward_tape)

        self.optimizer.zero_grad()

        for i in range(len(recorder.state_tape)):

            prob = net(recorder.state_tape[i])
            dist = Bernoulli(prob)
            loss = -dist.log_prob(recorder.action_tape[i]) * recorder.reward_tape[i]  # use original action
            loss.backward()

        self.optimizer.step()

    @staticmethod
    def _cal_reward(tape):
        """accumulate and adjust scores"""

        size = len(tape)
        for i in reversed(range(size-1)):
            tape[i] += tape[i+1] * GAMMA

        reward_mean = np.mean(tape)
        reward_std = np.std(tape)

        for i in range(size):
            tape[i] = (tape[i] - reward_mean) / reward_std


def train():
    """running environment"""

    recorder = Recorder()
    pol_grad = PolGrad()
    batch_steps = []
    row = 0

    for episode in range(1000):

        # use next_state to be clear
        next_state = env.reset()

        for step in count():
            state = torch.from_numpy(next_state).float()

            probs = net(state)
            dist = Bernoulli(probs)
            action = dist.sample()

            next_state, reward, done, __ = env.step(int(action.item()))
            recorder.record(state, action, reward)

            if done:
                break

        #     if step == 499:
        #         row += 1
        #
        #     if row >= 20:
        #         print("20 in a row")
        #         return
        #
        batch_steps.append(step)
        logger.info(f'Episode {episode} lasts {step} steps')
        # if (episode+1) % 50 == 0:
        #     logger.info(
        #         f'Episode {episode+1}\tMean {np.mean(batch_steps):3.0f}\
        #         Stdev {np.std(batch_steps):.0f}\
        #         499 {batch_steps.count(499)}'
        #     )
        #     batch_steps = []

        pol_grad.learn(recorder)
        recorder.reset()

    plt.plot(batch_steps)
    plt.show()


if __name__ == '__main__':
    train()
