from qlearning4k.agents.agents_twenty48 import *
from qlearning4k.games.twenty48 import Twenty48
import os.path
from multiprocessing import Pool

def do_agent(i):
    def num_zeros_border(state):
        return num_zeros(state) + np_on_edge(state)

    def all_heuristics(state):
        return num_zeros(state) + np_on_edge(state) + smooth(state) + snake(state)

    agents = [expecti(all_heuristics), expecti(num_zeros_border), expecti(snake), expecti(smooth)]
    agents = [random_strategy, static_preference_strategy, highest_reward_strategy] + agents
    names = ['random_strategy', 'cyclic', 'greedy', 'all_h', 'num_zeros_border', 'snake', 'smooth']

    agent = agents[i]
    name = names[i]

    filename = '%s.csv' % name
    if not os.path.exists(filename):
        results = play(Twenty48(), agent)
        np.savetxt(filename, results, delimiter=',', header='score,moves,time,max_tile')

if __name__ == '__main__':
    names = ['random_strategy', 'cyclic', 'greedy', 'all_h', 'num_zeros_border', 'snake', 'smooth']
    pool = Pool()
    pool.map(do_agent, (i for i in range(len(names))))
