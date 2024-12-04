from agent.expert_agent import Expert_agent
from fundamental.board import GameState
from fundamental.coordinate import Player


def run_black_is_expert():
    agent = Expert_agent()
    game = GameState.new_game(5,9)
    while True:
        end, winner = game.game_over()
        if end:
            break
        if game.player == Player.black:
            move = agent.select_move(game)
        else:
