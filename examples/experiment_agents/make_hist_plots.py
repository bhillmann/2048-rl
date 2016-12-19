import os.path
import numpy as np

from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

plt.style.use('seaborn-deep')

names = ['cnn_rl', 'nn_rl', 'random_strategy', 'cyclic', 'greedy', 'all_h', 'num_zeros_border', 'snake', 'smooth']
plt_names = ['CNN Reinforcement Learner', 'NN Reinforcement Learner', 'Random', 'Cyclic', 'Greedy', 'Expectimax (All Heuristics)', 'Expectimax (Number Zeros & Border)', 'Expectimax (Snake)', 'Expectimax (Smooth)']

random = np.loadtxt('random_strategy.csv', delimiter=',')
test_distribution = np.log2(random[:, 3])

plts = []
# header='score,moves,time,max_tile'
results = []
print('avg_time_per_move, max_score, max_tile, mean_tile, std_tile, p_512, p_1024, p_2048, p-val')
for plt_name, name in zip(plt_names, names):
    filename = '%s.csv' % name
    if os.path.exists(filename):
        results = np.loadtxt(filename, delimiter=',')
        bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        plt.hist(np.vstack((test_distribution, np.log2(results[:, 3]))).T, bins, label=['Random', plt_name])
        plt.legend(loc='upper left')
        # Label x and y axis
        plt.xlabel('Max Tile')
        plt.ylabel('Number of Games')
        # Plot-title
        plt.title(plt_name)
        plt.xticks(bins, [2**i for i in bins])
        plt.savefig('%s.png' % name)
        plt.close()
        avg_time_per_move = np.mean(results[:, 2]/results[:, 1])
        max_score = np.max(results[:, 0])
        max_tile = np.max(results[:, 3])
        mean_tile = np.mean(results[:, 3])
        std_tile = np.std(results[:, 3])
        p_512 = np.sum(results[:, 3] >= 512)
        p_1024 = np.sum(results[:, 3] >= 1024)
        p_2048 = np.sum(results[:, 3] >= 2048)
        print('%s, %.5f, %d, %d, %.2f, %.2f, %d, %d, %d, %.5f' % (plt_name, avg_time_per_move, max_score, max_tile, mean_tile, std_tile, p_512, p_1024, p_2048, mannwhitneyu(test_distribution, np.log2(results[:, 3])).pvalue))

