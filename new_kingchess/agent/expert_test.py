from agent.expert_agent import Expert_agent
from fundamental.board import GameState
from fundamental.coordinate import Move, Point

expert = Expert_agent()


game = GameState.new_game(5,9)




moves = expert.score_moves(game)

print(moves)
game = game.apply_move(Move(Point(2,2), Point(4,4)))
moves = expert.score_moves(game)

print(moves)

