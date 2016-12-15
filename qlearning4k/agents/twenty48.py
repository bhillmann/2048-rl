"""Algorithms and strategies to play 2048 and collect experience."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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

def make_greedy_strategy(get_q_values, verbose=False):
  """Makes greedy_strategy."""

def greedy_strategy(state, actions):
  """Strategy that always picks the action of maximum Q(state, action)."""
  q_values = get_q_values(state)
  if verbose:
    print("State:")
    print(state)
    print("Q-Values:")
    for action, q_value, action_name in zip(range(4), q_values, ACTION_NAMES):
      not_available_string = "" if action in actions else "(not available)"
      print("%s:\t%.2f %s" % (action_name, q_value, not_available_string))
  sorted_actions = np.argsort(q_values)
  action = [a for a in sorted_actions if a in actions][-1]
  if verbose:
    print("-->", ACTION_NAMES[action])
  return action

return greedy_strategy


def make_epsilon_greedy_strategy(get_q_values, epsilon):
  """Makes epsilon_greedy_strategy."""

  greedy_strategy = make_greedy_strategy(get_q_values)

  def epsilon_greedy_strategy(state, actions):
    """Picks random action with prob. epsilon, otherwise greedy_strategy."""
    do_random_action = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
    if do_random_action:
      return random_strategy(state, actions)
    return greedy_strategy(state, actions)

  return epsilon_greedy_strategy


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
    if verbose:
      print("Score:", game._score)
      game.print_state()

    old_state = state
    next_action = strategy(
        old_state, range(4) if allow_unavailable_action
                            else game.available_actions())

    if game.is_action_available(next_action):

      reward = game.do_action(next_action)
      state = game.state().copy()
      game_over = game.game_over()

      if verbose:
        print("Action:", ACTION_NAMES[next_action])
        print("Reward:", reward)

      experiences.append(Experience(old_state, next_action, reward, state,
                                    game_over, False, game.available_actions()))

    else:
      experiences.append(Experience(state, next_action, 0, state, False, True,
                                    game.available_actions()))

  if verbose:
    print("Score:", game.score())
    game.print_state()
    print("Game over.")

  return game.score(), experiences