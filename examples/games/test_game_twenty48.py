from qlearning4k.games import Twenty48

game_matrix = Twenty48()
print(game_matrix.grid)
for i in range(10):
    game_matrix.move(0)
    print(game_matrix.get_score())
print(game_matrix.grid)
