import copy
import datetime
import json
import random
import time
import uuid

import numpy as np
from flask import jsonify

from agent.alpha_beta import Alpha_beta
from agent.expert_agent import Expert_agent
from agent.random_agent import Random_agent
from fundamental.board import game_model, game_model_to_dict, GameState
from fundamental.utils import move_2_move_str, move_str2_move
from httpfront.exts import redis
from httpfront.models.models import UserModel
from httpfront.config import ROOM, TTL
from httpfront.services.userService import get_users
from socket_transfer_data.json_gamestate import json_state
from socket_transfer_data.server import send_request


def set_room(message):
    # print(message)

    user = None
    for user_redis in get_users():
        if user_redis['id'] == message['data']['from']:
            user_redis['status'] = 'chessing'
            user = user_redis

# print(user)
    room = copy.deepcopy(ROOM)

    room['id'] = 'room' + str(uuid.uuid4())

    room['host']['id'] = message['data']['from']

    room['host']['name'] = user['name']

    room['host']['color'] = message['data']['color']

    room['host']['status'] = user['status']  # chessing

    # print(room)

    room['time'] = time.time()

    redis.setex(room['id'], 150, json.dumps(room))

    room_msg = get_room(room['id'])

    # print(room_msg)
    # hgetall
    return room_msg


def set_room_vs_ai(message):
    user = None
    for user_redis in get_users():
        if user_redis['id'] == message['data']['from']:
            user_redis['status'] = 'chessing'
            user = user_redis

    # print(user)

    room = copy.deepcopy(ROOM)

    room['id'] = 'room' + str(uuid.uuid4())

    room['host']['id'] = message['data']['from']

    room['host']['name'] = user['name']

    room['host']['color'] = message['data']['color']

    room['host']['status'] = user['status']  # chessing

    if message['data']['type'] == 0:
        AI = UserModel.query.filter_by(username="random").first()

    if message['data']['type'] == 1:
        AI = UserModel.query.filter_by(username="expert").first()
    if message['data']['type'] == 2:
        AI = UserModel.query.filter_by(username="Alpha beta").first()
    if message['data']['type'] == 3:
        AI = UserModel.query.filter_by(username="resnet").first()
    if message['data']['type'] == 4:
        AI = UserModel.query.filter_by(username="vit").first()

    room['challenger']['id'] = AI.id
    room['challenger']['name'] = AI.username
    room['challenger']['color'] = 1 if message['data']['color'] == 0 else 0
    room['challenger']['status'] = user['status']
    room['challenger']['role'] = 'challenger'
    room['challenger']['ready'] = True

    room['time'] = time.time()

    redis.setex(room['id'], 120, json.dumps(room))
    # hgetall
    return get_room(room['id'])


def get_room(rid):
    # result = []
    # for room in [eval(i.decode('utf-8')) for i in redis.lrange('room', 0, -1)]:
    #     # print(time.time() - room['time'])
    #     if (time.time() - room['time']) > 120:
    #         redis.lrem('room', 0, json.dumps(room))
    #     else:
    #         result.append(room)

    return json.loads(redis.get(rid).decode('utf-8'))


def l_room(message):
    # {'code': 7, 'data': {'rid': 'room1398', 'id': 7}}

    # print(message)

    # user = None
    # for user_redis in get_users():
    #     if user_redis['id'] == message['data']['id']:
    #         user = user_redis

    room = get_room(message['data']['rid'])

    if room['challenger']['id'] == message['data']['id']:
        # 挑战者退出房间
        room['challenger'] = copy.deepcopy(ROOM['challenger'])
        # 修改用户的status
        redis.setex(room['id'], redis.ttl(room['id']), json.dumps(room))

        return get_room(message['data']['rid'])

    else:
        for spec in room['spectators']:
            if spec['id'] == message['data']['id']:
                room['spectators'].remove(spec)
        redis.setex(room['id'], float(redis.ttl(room['id'])), json.dumps(room))
        return get_room(message['data']['rid'])


def get_rooms():
    result = []
    for i in redis.keys():
        if i.decode('utf-8').startswith('room'):
            result.append(get_room(i.decode('utf-8')))
    return result


def del_room(message):
    # {'code': 8, 'data': 'room2980'}
    # for room in get_rooms():
    #     if room['id'] == message['data']:
    #         redis.lrem('room', 0, json.dumps(room))
    #         break
    return redis.delete(message['data'])


def enter_room(message):
    # {'code': 6, 'data': {'rid': 'room9277', 'from': 8, 'role': 'challenger'}}
    print(message)
    # print('加入房间')

    user = None
    for user_redis in get_users():
        if user_redis['id'] == message['data']['from']:
            user = user_redis
    if message['data']['role'] == 'challenger':
        room = get_room(message['data']['rid'])

        # print(room)
        room['challenger']['id'] = message['data']['from']

        room['challenger']['name'] = user['name']

        room['challenger']['color'] = 1 if room['host']['color'] == 0 else 0

        room['challenger']['status'] = 'chessing'
        room['challenger']['role'] = 'challenger'

        redis.setex(room['id'], redis.ttl(room['id']), json.dumps(room))

        return get_room(room['id'])
    if message['data']['role'] == 'spectator':
        room = get_room(message['data']['rid'])
        # print(room)
        # redis.lpush(room['spectators'], 0, user['id'])
        room['spectators'].append(user)
        print(room)
        redis.setex(room['id'], redis.ttl(room['id']), json.dumps(room))
        return get_room(room['id'])


def set_ready(message):
    # {'code': 14, 'data': {'rid': 'room4882', 'id': 8}}
    room = get_room(message['data']['rid'])
    if room['id'] == message['data']['rid']:
        if message['data']['id'] == room['host']['id']:
            room['host']['ready'] = True
            if room['challenger']['ready']:
                redis.setex(room['id'], 60 * 30, json.dumps(room))
                return get_room(room['id'])
            else:
                redis.setex(room['id'], redis.ttl(room['id']), json.dumps(room))
                return get_room(room['id'])

        elif message['data']['id'] == room['challenger']['id']:
            room['challenger']['ready'] = True
            if room['host']['ready']:
                redis.setex(room['id'], 60 * 30, json.dumps(room))
                return get_room(room['id'])
            else:
                redis.setex(room['id'], redis.ttl(room['id']), json.dumps(room))
                return get_room(room['id'])


def add_room_step(message):
    # {'code': 15, 'data': {'rid': 'room2162', 'state': {
    #    'board': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1, 0, -1, 0], [0, -1, 1, -1, 0],
    #              [0, -1, -1, -1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'move': '2242',
    #    'eat': '32', 'player': 1, 'winner': 0, 'play_out': 1}, 'name': 7}}
    room = get_room(message['data']['rid'])
    if message['data']['state']['move'] is not None and len(message['data']['state']['move']) == 2:
        room['steps'] += message['data']['state']['move'] + 'ee'
    else:
        room['steps'] += message['data']['state']['move']
    # print(room)
    redis.setex(room['id'], redis.ttl(room['id']), json.dumps(room))
    # return get_room(room['id'])


def remove_room_step(rid):  # require rid
    room = get_room(rid)
    if len(room['steps']) >= 4:
        room['steps'] = room['steps'][:-4]
    # print(room)
    redis.setex(room['id'], redis.ttl(room['id']), json.dumps(room))


def room_steps_2_state(move_str):
    game = GameState.new_game(5, 9)

    for i in range(0, len(move_str), 4):
        move = move_str2_move(move_str[i:i + 4])
        game = game.apply_move(move)

    return game_model_to_dict(game)


def ai_move(game, room):
    move_msg = {}
    if room['challenger']['name'] == 'random':
        # random
        ramdom = Random_agent()
        move = ramdom.select_move(game)
        game = game.apply_move(move)

    elif room['challenger']['name'] == 'expert':
        # expert
        expert = Expert_agent()
        move = expert.select_move(game)
        game = game.apply_move(move)

    elif room['challenger']['name'] == 'Alpha beta':
        # alpha beta
        alpha = Alpha_beta()
        move = alpha.select_move(game)
        game = game.apply_move(move)

    elif room['challenger']['name'] == 'resnet':
        json_game = json_state(game)
        # action = send_request('127.0.0.1', 8900, json_game)
        # 117.72.75.113
        action = send_request('10.122.7.125', 8900, json_game)
        if isinstance(action, np.int64) or isinstance(action, int):
            game = game.apply_move(game.a_trans_move(action))
        else:
            game = game.apply_move(action)
    elif room['challenger']['name'] == 'vit':
        json_game = json_state(game)
        # action = send_request('127.0.0.1', 9000, json_game)
        action = send_request('10.122.7.125', 9000, json_game)
        if isinstance(action, np.int64) or isinstance(action, int):
            game = game.apply_move(game.a_trans_move(action))
        else:
            game = game.apply_move(action)

    state = game_model_to_dict(game)
    move_msg['rid'] = room['id']
    move_msg['state'] = state
    move_msg['name'] = room['challenger']['name']
    return move_msg


import threading
