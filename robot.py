import random

import numpy as np


class Robot(object):

    def __init__(self):
        self.num_actions = 4
        self.epsilon = 0.1
        self.alpha = 0.9
        self.gamma = 0.9

        self.q = {}
        self.state = "0"
        self.q[self.state] = [0.0, 0.0, 0.0, 0.0]
        self.action = 0
        self.total_reward = 0

    def do_action(self):
        if self.state not in self.q:
            self.q[self.state] = [0.0, 0.0, 0.0, 0.0]
        if random.uniform(0, 1) < self.epsilon:
            self.action = random.choice(range(4))
        else:
            self.action = np.argmax(self.q[self.state])
        return self.action

    def update(self, reward, points):
        self.total_reward += reward
        new_state = ''
        for point in points:
            new_state += str(point)
        if new_state not in self.q:
            self.q[new_state] = [0.0, 0.0, 0.0, 0.0]
        old_value = self.q[self.state][self.action]
        next_max = np.max(self.q[new_state])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q[self.state][self.action] = new_value

        self.state = new_state
