import numpy as np
import random

from .game import Game


class Twenty48(Game):
    def __init__(self, grid_size=4):
        """ Constructs a new Model object """
        self.gameover = False
        self.grid_size = (grid_size, grid_size)
        self.grid = np.zeros(shape=self.grid_size).astype('int')
        self.new_block()
        self.new_block()

    @property
    def nb_actions(self):
        return 4

    @property
    def name(self):
        return "2048"

    def get_possible_actions(self):
        return [i for i in range(4) if self.move_dir(i)]

    def get_state(self):
        return self.grid

    def get_score(self):
        if self.gameover:
            return -1
        else:
            return np.max(self.grid)/2048

    def is_won(self):
        return self.get_score() >= 2048

    def reset(self):
        """
        Reset the game
        """
        self.gameover = False
        print(np.max(self.grid))
        print(self.grid)
        self.grid = np.zeros(shape=self.grid_size).astype('int')
        self.new_block()
        self.new_block()

    def move_dir(self, direction):
        if direction == 0:  # UP
            a, b = (-1, 0)
        elif direction == 1:  # RIGHT
            a, b = (1, 1)
        elif direction == 2:  # DOWN
            a, b = (1, 0)
        elif direction == 3:  # LEFT
            a, b = (-1, 1)
        else:
            return False

        prev = self.grid.copy()
        prev = self.shift(a, b, prev)
        if not (prev == self.grid).all():
            return True, prev
        else:
            return False, []

    def play(self, direction):
        self.move(direction)

    def move(self, direction):
        moved, temp_grid = self.move_dir(direction)
        if moved:
            self.grid = temp_grid
            if self.is_over():
                self.gameover = True
            self.new_block()
            return True
        else:
            return False

    def has_moves(self):
        if (self.grid[1:, :] == self.grid[:-1, :]).any():
            return True
        if (self.grid[:, 1:] == self.grid[:, :-1]).any():
            return True
        return False

    def is_over(self):
        return not self.has_moves() and len(np.where(self.grid == 0)[0]) == 0

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

    def new_block(self):
        block = 2
        if np.random.uniform(0, 1) > .9:
            block = 4
        rows, cols = np.where(self.grid == 0)
        if len(rows) == 0:
            return
        row, col = random.choice(zip(rows, cols))
        self.grid[row, col] = block


def main():
    game_matrix = Game2048()
    print(game_matrix.grid)
    for i in range(10):
        game_matrix.move(0)
    print(game_matrix.grid)

if __name__ == "__main__":
    main()
