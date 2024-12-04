import pickle

import numpy as np
import tqdm

from a3c.collect_data import get_equi_data
from agent.expert_agent import Expert_agent
from fundamental.board import GameState
from net.encoder import encoder_board


def main_simple():
    play_data = []
    row_size = 5
    col_size = 9
    game = GameState.new_game(row_size, col_size)
    # bot1 = Random_agent()
    bot2 = Expert_agent()
    states, mcts_probs, current_players = [], [], []

    while True:

        end, winner = game.game_over()
        if end:
            winners_z = np.zeros(len(current_players))
            winners_z[np.array(current_players) == winner] = 1.0
            winners_z[np.array(current_players) != winner] = -1.0

            play_data = zip(states, mcts_probs, winners_z)
            play_data = list(play_data)[:]
            play_data = get_equi_data(play_data)
            #play_datas.extend(play_data)
            
            return play_data

        move = bot2.select_move(game)
        states.append(encoder_board(game))
        move_probs = np.zeros((1125))
        move_probs[game.move_2_action(move)] = 1
        mcts_probs.append(move_probs)
        current_players.append(game.player)
        game = game.apply_move(move)

    return winner

if __name__ == '__main__':
    play_datas = []
    for i in tqdm.tqdm(range(50000)):
        play_data = main_simple()
        play_datas.extend(play_data)
    with open('./expert_data.pkl', 'ab') as f:
        pickle.dump(play_datas, f)
