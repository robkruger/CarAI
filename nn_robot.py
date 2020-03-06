from __future__ import division

import joblib
import numpy as np
import random

import pygame
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

from game import Game

import pandas as pd
from operator import add


class DQN:
    def __init__(self):
        self.memory = deque(maxlen=1000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=6))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=6, activation='softmax'))
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    def act(self, state, games):
        self.epsilon = 80 - games
        if np.random.randint(0, 200) < self.epsilon:
            return random.randrange(6)
        return np.argmax(self.model.predict(state.reshape(1, 6))[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array(next_state.reshape(1, 6)))[0])
            target_f = self.model.predict(state.reshape(1, 6))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array(state.reshape(1, 6)), target_f, epochs=1, verbose=0)

    def train_short_memory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 6)))[0])
        target_f = self.model.predict(state.reshape((1, 6)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 6)), target_f, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():
    pygame.init()
    visualize = False
    trial = 0
    while 1:
        cur_state = np.array([0, 0, 0, 0, 0, 0.0])
        g = Game((1024, 768), None)
        done = False
        trial += 1
        print("Next trial:", trial)
        steps = 0
        while not done:
            steps += 1
            for event in pygame.event.get():
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        visualize = not visualize
                    if event.key == pygame.K_RETURN:
                        dqn_agent.model.save('model')
                        dqn_agent.target_model.save('target_model')
                        joblib.dump(dqn_agent.memory, "memory.sav", 2)
                        print("Saved models")
                    if event.key == pygame.K_l:
                        dqn_agent.model = load_model('model')
                        dqn_agent.target_model = load_model('target_model')
                        dqn_agent.memory = joblib.load("memory.sav")
                        print("Loaded models")

            action = dqn_agent.act(cur_state, trial)
            new_state, reward, done, cr_ch = g.parse_events(action)

            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.train_short_memory(cur_state, action, reward, new_state, done)

            cur_state = new_state
            if visualize:
                g.draw()

            if cr_ch:
                steps = 0

            if steps > 200:
                print("More than 200 steps, probably stuck. Resetting.")
                break

        dqn_agent.replay_new(dqn_agent.memory)  # internally iterates default (prediction) model
        dqn_agent.target_train()  # iterates target model

        if trial % 10 == 0:
            dqn_agent.model.save('model')
            dqn_agent.target_model.save('target_model')
            joblib.dump(dqn_agent.memory, "memory.sav", 2)
            print("Saved models")


if __name__ == "__main__":
    main()