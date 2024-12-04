import json

import os
import sys
import threading

cur_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(cur_path)
from flask import Flask, session, redirect, url_for
from flask import jsonify
from flask import request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, send, emit
from httpfront.exts import redis

# from geventwebsocket.websocket import WebSocket
# from gevent.pywsgi import WSGIServer
# from geventwebsocket.handler import WebSocketHandler

import eventlet
import agent
from fundamental.board import GameState, game_model
from fundamental.utils import coords_from_point, print_board, move_2_move_str

from fundamental.utils import point_from_coords
from httpfront import config
from httpfront.exts import db
from httpfront.exts import socketio
from httpfront.views.hall import hall_bp
from httpfront.views.user import user_bp
from httpfront.views.common import common_bp
from httpfront.models.models import UserModel
from flask_migrate import Migrate

# __all__ = [
#     'get_web_app',
# ]

here = os.path.dirname(__file__)
static_path = os.path.join(here, 'static')
app = Flask(__name__, static_folder=static_path, static_url_path='/static')
# app.config['SECRET_KEY'] = 'secret_key'
CORS(app)

app.config.from_object(config)

db.init_app(app)
redis.init_app(app)
migrate = Migrate(app, db)

app.register_blueprint(user_bp)
app.register_blueprint(hall_bp)
app.register_blueprint(common_bp)
# socketio = SocketIO(logger=True, engineio_logger=True)

socketio.init_app(app, cors_allowed_origins='*')  # cors_allowed_origins='*'

def get_web_app(bot_map):
    """Create a flask application for serving bot moves.

    The bot_map maps from URL path fragments to Agent instances.

    The /static path will return some static content (including the
    jgoboard JS).

    Clients can get the post move by POSTing json to
    /select-move/<bot name>

    Example:

    >>> myagent = agent.RandomBot()
    >>> web_app = get_web_app({'random': myagent})
    >>> web_app.run()

    Returns: Flask application instance
    """

    @app.route('/new_game/<bot_name>', methods=['POST'])
    def new_game(bot_name):
        content = request.get_json()
        player = content['player']
        if player == '1':
            return jsonify({
                'result': '你当前为国王，请走棋！！！'
            })
        elif player == '-1':
            game = GameState.new_game(5, 9)
            end, winner = game.game_over()
            if end:
                return jsonify({
                    'gameover': end,
                    'winner': winner
                    # 'diagnostics': bot_map.diagnostics()
                })
            bot_agent = bot_map[bot_name]
            #
            bot_move = bot_agent.select_move(game)
            bot_move_str = move_2_move_str(bot_move)
            return jsonify({
                'bot_move': bot_move_str,
            })

    @app.route('/select-move/<bot_name>', methods=['POST'])
    def select_move(bot_name):
        content = request.get_json()
        game_state = game_model(content)

        # return jsonify({
        #     'hello':1,
        # })
        end, winner = game_state.game_over()
        if end:
            return jsonify({
                'gameover': end,
                'winner': winner
                # 'diagnostics': bot_map.diagnostics()
            })
        #     # board = content['board']
        #
        #     # Replay the game up to this point.
        #     # print(content)
        #     # print(type(content))
        #     # print_board(game_state.board)
        #     # print(len(content['moves']))
        bot_agent = bot_map[bot_name]
        #
        bot_move = bot_agent.select_move(game_state)
        #     # print(bot_name)
        #     # print_board(game_state.board)
        #     # print(game_state.board.get_grid)
        #     # print(bot_map)
        #     # print(bot_map[bot_name])
        #     # bot_agent = bot_map[bot_name]
        #
        #     # bot_move = bot_agent.select_move_stage1(game_state)
        #
        #     # row = int(bot_move.point.row)
        #     # col = int(bot_move.point.col)
        #     # point = Point(row, col)
        bot_move_str = move_2_move_str(bot_move)
        print(f'bot_move_str:{bot_move_str}')
        #
        #     # print('bot_move_str')
        #     # print(bot_move_point)
        #
        return jsonify({
            'bot_move': bot_move_str,
            # 'diagnostics': bot_map.diagnostics()
        })

    #
    # # @app.route('/select-move/<error>', methods=['GET'])
    # # def process_error(error):
    # #     return jsonify({
    # #         'error': error,
    # #     })

    return app


# @app.before_request
# def check_session():
#     if not session.get('user_id') and request.endpoint not in ['login']:
#         return redirect(url_for('login'))


if __name__ == '__main__':
    from agent import random_agent
    from net.mcts_alphazreo import MCTSPlayer
    from net.policy_value_net_pytorch import PolicyValueNet

    myagent = random_agent.Random_agent()
    # policy_value_net_current = PolicyValueNet(model_file='E:/new_model_4_19/get_muc_model_5_28/current.pt')
    # mcts_current = MCTSPlayer(policy_value_net_current.policy_value_fn, c_puct=5, n_playout=1200)
    # app = get_web_app({'random': myagent, 'mcts': mcts_current})
    # app.run(host='0.0.0.0', port=8888, debug=True)

    # handle_notifications()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
