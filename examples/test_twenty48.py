from keras.models import Sequential
from keras.initializations import normal
from keras.layers import Flatten, Dense, Convolution2D
from qlearning4k.games import Twenty48
from keras.optimizers import *
from qlearning4k import Agent
from qlearning4k import ExperienceReplay

grid_size = 4
hidden_size = 256
nb_frames = 1
memory = ExperienceReplay(100000, fast=None)

twenty48 = Twenty48(grid_size=grid_size)

model = Sequential()
# model.add(Convolution2D(32, 2, 2, activation='relu', init=lambda shape, name: normal(shape, scale=0.01, name=name), input_shape=(nb_frames, grid_size, grid_size)))
# model.add(Convolution2D(16, 2, 2, activation='relu', init=lambda shape, name: normal(shape, scale=0.01, name=name)))
# model.add(Convolution2D(8, 2, 2, activation='relu',  init=lambda shape, name: normal(shape, scale=0.01, name=name)))
model.add(Flatten(input_shape=(nb_frames, 17)))
model.add(Dense(hidden_size, init=lambda shape, name: normal(shape, scale=0.01, name=name), activation='relu'))
model.add(Dense(hidden_size, init=lambda shape, name: normal(shape, scale=0.01, name=name), activation='softmax'))
model.add(Dense(twenty48.nb_actions))
model.compile(Adam(lr=1e-6), "mse")

agent = Agent(model=model, memory=memory, nb_frames=nb_frames)
agent.train(twenty48, batch_size=32, nb_epoch=10000000, observe=1000, epsilon=[0., 0.])
agent.play(twenty48, visualize=False)
