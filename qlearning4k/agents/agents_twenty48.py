"""Algorithms and strategies to play 2048 and collect experience."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import itertools
import numpy as np
from qlearning4k.games import Twenty48


def random_strategy(_, actions):
    """Strategy that always chooses actions at random."""
    return np.random.choice(actions)


def static_preference_strategy(_, actions):
    """Always prefer left over up over right over top."""
    return min(actions)


def highest_reward_strategy(state, actions):
    """Strategy that always chooses the action of highest immediate reward.
    If there are any ties, the strategy prefers left over up over right over down.
    """

    sorted_actions = np.sort(actions)[::-1]
    rewards = map(lambda action: state.copy().do_action(action),
                sorted_actions)
    action_index = np.argsort(rewards, kind="mergesort")[-1]
    return sorted_actions[action_index]


def play(game, strategy, verbose=False, allow_unavailable_action=False):
    """Plays a single game, using a provided strategy.
    Args:
    strategy: A function that takes as argument a state and a list of available
    actions and returns an action from the list.
    allow_unavailable_action: Boolean, whether strategy is passed all actions
    or just the available ones.
    verbose: If true, prints game states, actions and scores.
    Returns:
    score, experiences where score is the final score and experiences is the
    list Experience instances that represent the collected experience.
    """
    state = game.copy()
    game_over = game.is_over()

    while not game_over:
        action = strategy(state, state.get_possible_actions())
        print(state.grid)
        state.move(action)
        game_over = state.is_over()

    return 2**np.max(state.grid)


def fitness(state):
    """
    Returns the heuristic value of b

    Snake refers to the "snake line pattern" (http://tinyurl.com/l9bstk6)
    Here we only evaluate one direction; we award more points if high valued tiles
    occur along this path. We penalize the board for not having
    the highest valued tile in the lower left corner
    """

    snake = []
    for i in range(4):
        snake.extend(reversed(state.grid[:, i]) if i % 2 == 0 else state.grid[:, i])

    m = max(snake)
    return sum(x / 10 ** n for n, x in enumerate(snake)) - \
           math.pow((state.grid[3, 0] != m) * abs(state.grid[3, 0] - m), 2)

def expecti(state, actions, d=7):
    """
    Performs expectimax search on a given configuration to
    specified depth (d).

    Algorithm details:
       - if the AI needs to move, make each child move,
         recurse, return the maximum fitness value
       - if it is not the AI's turn, form all
         possible child spawns, and return their weighted average
         as that node's evaluation
    """

    def alpha_beta_search(state, d, move=False):
        if d == 0 or state.is_over():
            return fitness(state)

        alpha = fitness(state)
        if move:
            for action in state.get_possible_actions():
                temp = state.copy()
                temp.move(action)
                return max(alpha, alpha_beta_search(temp, d - 1))
        else:
            zeros = [(i,j) for i, j in itertools.product(range(4), range(4)) if state.grid[i][j] == 0]
            for i, j in zeros:
                state2 = state.copy()
                state.grid[i, j] = 2
                state2.grid[i, j] = 4
                alpha += .9*alpha_beta_search(state, d-1, move=True)/len(zeros) + .1*alpha_beta_search(state2, d-1, move=True)/len(zeros)
        return alpha

    best_action = 0
    best_alpha = -np.inf
    for action in actions:
        temp = state.copy()
        temp.move(action)
        alpha = alpha_beta_search(temp, 5)
        if alpha > best_alpha:
            best_action = action
            best_alpha = alpha
    return best_action

print(play(Twenty48(), expecti))
