from fundamental.board import GameState
from fundamental.coordinate import Point, Player, Move
from fundamental.utils import print_board, list_from_board




if __name__ == '__main__':
    game = GameState.new_game(5, 9)
    game.board.get_grid()[Point(0,0)] = Player.black

    game.apply_move(Move(Point(0,0), Point(2,0)))

