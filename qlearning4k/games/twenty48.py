import numpy as np
import random
from scipy import signal

from .game import Game


class Twenty48(Game):
    """
    Game state are represented as shape (4, 4) numpy arrays whose entries are 0 for empty fields and ln2(values) for
    any tiles.
    """

    def __init__(self, grid_size=4, state=None, initial_score=0, num_moves=0):
        """Init the Game object.
        Args:
          state: Shape (4, 4) numpy array to initialize the state with. If None,
              the state will be initialized with with two random tiles (as done
              in the original game).
          initial_score: Score to initialize the Game with.
        """
        self._score = initial_score
        self._reward = 0.
        self.game_over = None
        self._num_moves = num_moves


        if state is None:
            self.grid = None
            self.grid_size = (grid_size, grid_size)
            self.reset()
        else:
            self.grid_size = state.shape
            self.grid = state
            self.game_over = self._is_over()

    def copy(self):
        return Twenty48(state=np.copy(self.grid), initial_score=self._score, num_moves=self._num_moves)

    @property
    def nb_actions(self):
        return 4

    @property
    def name(self):
        return "2048"

    def get_possible_actions(self):
        return [action for action in range(self.nb_actions) if self.is_action_available(action)]

    def is_action_available(self, action):
        """Determines whether action is available.
        That is, executing it would change the state.
        """

        temp_state = np.rot90(self.grid, action)
        return self._is_action_available_left(temp_state)

    def _is_action_available_left(self, state):
        """Determines whether action 'Left' is available."""

        # True if any field is 0 (empty) on the left of a tile or two tiles can
        # be merged.
        for row in xrange(self.grid_size[0]):
            has_empty = False
            for col in xrange(self.grid_size[1]):
                has_empty |= state[row, col] == 0
                if state[row, col] != 0 and has_empty:
                    return True
                if (state[row, col] != 0 and col > 0 and
                            state[row, col] == state[row, col - 1]):
                    return True

        return False

    def get_state(self):
        x = np.sum(self.grid, axis=1).flatten()
        y = np.sum(self.grid, axis=0).flatten()
        z = signal.convolve2d(self.grid, np.array([[1, 1], [1, 1]]), mode='valid').flatten()
        return np.hstack([x, y, z])
        # return self.grid

    def get_score(self):
        if self._reward == 0:
            return 0.
        else:
            # print(np.log2(self._reward)/11.)
            return np.log2(self._reward)/11.
        # if self._reward > 0:
        #     return np.log2(self._reward)
        # return 0
        # return np.max(self.grid)/11.

    def is_won(self):
        return np.max(self.grid) >= 11

    def reset(self):
        """
        Reset the game
        """
        if not self.grid == None:
            print(2**np.max(self.grid))
        self._num_moves = 0
        self.game_over = False
        self.grid = np.zeros(shape=self.grid_size).astype('int')
        self.add_random_tile()
        self.add_random_tile()

    def do_action(self, state, action):
        temp_state = np.rot90(state, action)
        reward = self._do_action_left(temp_state)
        temp_state = np.rot90(temp_state, -action)

        return temp_state, reward

    def _do_action_left(self, state):
        """Executes action 'Left'."""

        reward = 0

        for row in range(4):
            # Always the rightmost tile in the current row that was already moved
            merge_candidate = -1
            merged = np.zeros((4,), dtype=np.bool)

            for col in range(4):
                if state[row, col] == 0:
                    continue

                if (merge_candidate != -1 and
                        not merged[merge_candidate] and
                            state[row, merge_candidate] == state[row, col]):
                    # Merge tile with merge_candidate
                    state[row, col] = 0
                    merged[merge_candidate] = True
                    state[row, merge_candidate] += 1
                    reward += 2 ** state[row, merge_candidate]

                else:
                    # Move tile to the left
                    merge_candidate += 1
                    if col != merge_candidate:
                        state[row, merge_candidate] = state[row, col]
                        state[row, col] = 0

        return reward

    def play(self, action):
        self.move(action)

    def move(self, action):
        if self.is_action_available(action):
            temp_state, reward = self.do_action(np.copy(self.grid), action)
            self._score += reward
            self._reward = reward
            self.grid = temp_state
            self.add_random_tile()
            if self._is_over():
                self.game_over = True
            self._num_moves += 1
            return True
        else:
            return False

    def is_over(self):
        return self.game_over

    def _is_over(self):
        for action in xrange(self.nb_actions):
            if self.is_action_available(action):
                return False
        return True

    def shift(self, way, axis, grid):
        for y in xrange(grid.shape[axis]):
            curr = grid[y, :] if axis == 0 else grid[:, y]
            curr = sorted(curr, key=lambda x: way * (x != 0))
            range_ = range(len(curr))
            x = 0 if way == -1 else len(curr) - 1
            while True:
                next_ = x - way
                if next_ in range_:
                    if curr[x] == 0 and curr[next_] == 0:
                        x -= way
                    elif curr[x] == 0:
                        curr[x] = curr[next_]
                        curr[next_] = 0
                        x += way
                    elif curr[next_] == curr[x]:
                        curr[x] *= 2
                        curr[next_] = 0
                    else:
                        x -= way
                else:
                    break
            if axis == 0:
                grid[y, :] = curr
            else:
                grid[:, y] = curr
        return grid

    def add_random_tile(self):
        """Adds a random tile to the grid. Assumes that it has empty fields."""
        x_pos, y_pos = np.where(self.grid == 0)
        assert len(x_pos) != 0
        empty_index = np.random.choice(len(x_pos))
        value = np.random.choice([1,2], p=[0.9, 0.1])
        self.grid[x_pos[empty_index], y_pos[empty_index]] = value
