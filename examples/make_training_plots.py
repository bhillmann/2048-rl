import os.path
import numpy as np
import numpy as np
from numpy import convolve
import matplotlib.pyplot as plt
import re


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


comp_re = re.compile(r"[^0-9]")

plt.style.use('seaborn-deep')

files = ['cnn_training_results.txt', 'nn_training_results.txt']

training_scores = []
for file in files:
    this_scores = []
    with open(file) as inf:
        for line in inf:
            line = line.rstrip().replace(" ", "").split('|')
            if len(line) == 6:
                score = line[-1]
                this_scores.append(np.int(re.sub(comp_re, "", score)))
    training_scores.append(this_scores)

y = np.array(training_scores).T
x = np.arange(10000)

# plt.plot(x, y)
# plt.show()

cnn = movingaverage(y[:, 0], 100)
nn = movingaverage(y[:, 1], 100)
random = [np.log2(53.35)]*len(cnn)
greedy = [np.log2(91.68)]*len(cnn)

Y = np.vstack((random, greedy, cnn, nn)).T

for name, y in zip(['Random', 'Greedy', 'CNN Reinforcement Learner', 'NN Reinforcement Learner'], [random, greedy, cnn, nn]):
    plt.plot(x[len(x) - len(cnn):], y, label=name)
plt.title('Running Average of Reinforcement Learners Over Training Games')
plt.ylabel('Average Max Tile')
plt.xlabel('Number Training Games')
plt.legend(loc='upper left')
plt.yticks([5, 6, 7, 8, 9], [2**i for i in [5, 6, 7, 8, 9]])
plt.savefig('running_avg_per_training_game.png')
