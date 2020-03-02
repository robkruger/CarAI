import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import numpy as np
from random import randint

from game import Game

# variables for limiting the number of games we play
total_number_of_games = 100
games_count = 0

# The neural network training data
x_train = np.array([])
y_train = np.array([])

really_huge_number = 1000

# How frequently we train the neural network
train_frequency = 2

# The actual neural network
model = Sequential()
model.add(Dense(1, input_dim=6, activation='sigmoid'))
model.add(Dense(6, activation='softmax'))
model.compile(Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

all_scores = []
average_scores = []
average_score_rate = 10
all_x, all_y = np.array([]), np.array([])


class Wrapper(object):

    def __init__(self):
        # Start the game
        g = Game((1024, 768), self)
        g.controlled_run(self)

    def control(self, values):
        global x_train
        global y_train

        global games_count

        global model

        # This is the function that is called by the game.
        # The values dict contains important information
        # that we will need to use to train and predict
        print(values)

        lines = []
        for _ in range(len(values['lines'])):
            if values['lines'][_] == 'n':
                lines.append(1)
            else:
                lines.append(values['lines'][_] / 100)
        lines.append(values['speed'] * 10)
        arr = np.array([lines])

        if values['score_increased'] == 1:
            x_train = np.append(x_train, [arr])
            y_train = np.append(y_train, [values['action']])

        # Let's ask for input
        # print ("Enter 1 for JUMP and 0 for DO_NOTHING")
        # action = int(input())

        # The prediction from neural network
        print(arr)
        prediction2 = model.predict(arr)
        prediction = model.predict_classes(arr)

        r = randint(0, 100)

        random_rate = 50 * (1 - games_count / 50)

        if r < random_rate:
            return np.argmax(prediction)
        else:
            return np.argmax(prediction2)

    def gameover(self):
        global games_count
        global x_train
        global y_train
        global model

        global all_x
        global all_y
        global all_scores
        global average_scores
        global average_score_rate

        games_count += 1

        # Printing x_train and y_train
        print(x_train)
        print(y_train)

        all_x = np.append(all_x, x_train)
        all_y = np.append(all_y, y_train)

        if games_count is not 0 and games_count % train_frequency is 0 and len(x_train) > 0:
            # Before training, let's make the y_train array categorical
            y_train = y_train - 1
            y_train_cat = to_categorical(y_train, num_classes=2)

            x_train.reshape((int(x_train.shape[0] / 6), 6))

            print(x_train)

            # Let's train the network
            model.fit(x_train, y_train_cat, epochs=50, verbose=1, shuffle=1)

            # Reset x_train and y_train
            x_train = np.array([])
            y_train = np.array([])

        if games_count >= total_number_of_games:
            # Let's exit the program now
            return

        # Let's start another game!
        g = Game((1024, 768), self)
        g.controlled_run(self)


if __name__ == '__main__':
    w = Wrapper()
