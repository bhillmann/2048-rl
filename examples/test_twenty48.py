from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D
from qlearning4k.games import Twenty48
from keras.optimizers import *
from qlearning4k import Agent
from qlearning4k import ExperienceReplay

grid_size = 4
hidden_size = 256
nb_frames = 4
memory = ExperienceReplay(50000, fast=None)

twenty48 = Twenty48(grid_size=grid_size)

model = Sequential()
model.add(Convolution2D(64, 2, 2, activation='relu', input_shape=(nb_frames, grid_size, grid_size)))
model.add(Convolution2D(32, 2, 2, activation='relu', input_shape=(nb_frames, grid_size, grid_size)))
model.add(Convolution2D(16, 2, 2, activation='relu', input_shape=(nb_frames, grid_size, grid_size)))
model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(twenty48.nb_actions))
model.compile(Adam(lr=.0001), "mse")

agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)
agent.train(twenty48, batch_size=32, nb_epoch=1000, observe=10, epsilon=[1, 0.1])
agent.play(twenty48)
