from keras.models import Sequential
from keras.initializations import normal
from keras.layers import Flatten, Dense, Convolution2D
from qlearning4k.games.twenty48 import Twenty48
from keras.optimizers import *
from qlearning4k.agent import Agent
from qlearning4k.memory import ExperienceReplay
from qlearning4k.agents.agents_twenty48 import play
import numpy as np

grid_size = 4
hidden_size = 128
nb_frames = 1
memory = ExperienceReplay(50000, fast=None)
nb_epoch = 20
observe = 10

twenty48 = Twenty48(grid_size=grid_size)

model = Sequential()
# model.add(Convolution2D(32, 2, 2, activation='relu', init=lambda shape, name: normal(shape, scale=0.01, name=name), input_shape=(nb_frames, grid_size, grid_size)))
# model.add(Convolution2D(16, 2, 2, activation='relu', init=lambda shape, name: normal(shape, scale=0.01, name=name)))
# model.add(Convolution2D(8, 2, 2, activation='relu',  init=lambda shape, name: normal(shape, scale=0.01, name=name)))
model.add(Flatten(input_shape=(nb_frames, grid_size, grid_size)))
model.add(Dense(hidden_size, init=lambda shape, name: normal(shape, scale=0.01, name=name), activation='relu'))
model.add(Dense(hidden_size, init=lambda shape, name: normal(shape, scale=0.01, name=name), activation='softmax'))
model.add(Dense(twenty48.nb_actions))
model.compile(Adam(lr=1e-6), "mse")


agent = Agent(model=model, memory=memory, nb_frames=nb_frames)
agent.train(twenty48, batch_size=32, nb_epoch=nb_epoch, observe=observe, epsilon=[0., 0.], checkpoint=10000)

def agent_play_wrapper(agent):
    def play_agent(state, actions):
        model = agent.model
        s = state.grid
        s = s.reshape(1, 1, 4, 4)
        q_prime = model.predict(s)[0]
        q_prime = q_prime.argsort()
        for i in q_prime:
            if i in actions:
                a = i
                break
        return i
    return play_agent

results = play(Twenty48(), agent_play_wrapper(agent))
np.savetxt('rl.csv', results, delimiter=',', header='score,moves,time,max_tile')
# agent.play(twenty48, visualize=False)
