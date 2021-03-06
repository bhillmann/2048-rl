from keras.models import Sequential
from keras.layers import *
from qlearning4k.games import Snake
from keras.optimizers import *
from qlearning4k import Agent

grid_size = 10
nb_frames = 5
nb_actions = 5

model = Sequential()
model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(nb_frames, grid_size, grid_size)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(nb_actions))
model.compile(RMSprop(), 'MSE')

snake = Snake(grid_size)
print(model.summary())

# agent = Agent(model=model, memory_size=-1, nb_frames=nb_frames)
# agent.train(snake, batch_size=64, nb_epoch=10000, gamma=0.8)
# agent.play(snake)
