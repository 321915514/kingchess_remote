import datetime
import json
import random
import threading

from flask_socketio import send, emit, join_room, leave_room
from flask import Blueprint, jsonify, session, request

from fundamental.board import GameState, game_model_to_dict, game_model
from httpfront.exts import socketio, db
from httpfront.config import ROOM
from httpfront.models.models import UserModel, sgfModel, winRateModel
# from httpfront.redis_expire_check import listen_for_expirations
from httpfront.services.userService import get_user_status, get_users
from httpfront.services.roomService import set_room, get_rooms, l_room, enter_room, del_room, set_ready, add_room_step, \
    get_room, set_room_vs_ai, ai_move, remove_room_step, room_steps_2_state
from httpfront.exts import redis
from httpfront.views.user import check_user_session

hall_bp = Blueprint('hall', __name__, url_prefix='/')

# 用于存储监听到的过期键
expired_keys = set()
expired_users = set()


def listen_for_expirations():
    redis.config_set('notify-keyspace-events', 'Ex')
    pubsub = redis.pubsub()
    pubsub.psubscribe('__keyevent@0__:expired')
    # print("Listening for expired keys...")

    for message in pubsub.listen():
        if message['type'] == 'pmessage':
            key_expire = message['data'].decode()
            if key_expire.startswith('room'):
                expired_keys.add(key_expire)
            else:
                expired_users.add(key_expire)


@socketio.on('connect')
def connect_msg():
    # users = get_users()
    # if users:
    #     for i in users:
    #         join_room(i['id'])
    # socketio.start_background_task(listen_for_expirations)

    # if check_user_session():

    threading.Thread(target=listen_for_expirations).start()

    # print('client connect')


# else:
#     emit('message', {"code": 22, "data": {"msg": "请先登录"}})


@socketio.on('disconnect')
def connect_msg():
    # print('disconnect')
    pass


@socketio.on('message')
def myevent(message):
    # complete 11 2,
    message = json.loads(message)
    if message['code'] == 20:
        # print(message)
        # ttl = redis.ttl(message['data'])
        # print(ttl)
        # if ttl < 0:
        #     emit('message', {"code": 20, "data": {"msg": True}})
        pass

    # print(message)
    if message['code'] == 11:
        # print('getplayers')
        users = get_users()
        # print(users)
        emit('message', {"code": 11, "data": users})
    elif message['code'] == 4:
        # print("get rooms")

        rooms = get_rooms()

        emit('message', {
            "code": 4,
            "data": rooms
        })

    elif message['code'] == 2:
        # hall chat
        # print(message)
        emit('message', {
            "code": 2,
            "data": {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from": message['data']['name'],
                "content": message['data']['input'],
            }
        }, broadcast=True)

    elif message['code'] == 5:
        # create room
        # print(message)

        if 'type' in message['data']:
            room_msg = set_room_vs_ai(message)

        else:
            room_msg = set_room(message)

        # print(room_msg)
        join_room(message['data']['from'])
        emit("message", {"code": 5,
                         "data": room_msg}, to=message['data']['from'])

        # leave_room(message['data']['from'])
        join_room(room_msg['id'])

        # emit('message', {
        #     "code": 9,
        #     "data": {
        #         "rid": message['data']['rid'],
        #         "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #         "from": '系统消息',
        #         "content": '',
        #     }
        # }, to=room_msg['id'])

        rooms = get_rooms()

        # print(rooms)

        emit('message', {
            "code": 4,
            "data": rooms
        }, broadcast=True)




    elif message['code'] == 3:
        emit("message", {
            "code": 3,
            "data": [{
                "time": "2020-03-31 14:57:00",
                "from": "123",
                "content": "666"
            }, {
                "time": "2020-03-31 15:54:38",
                "from": "5be6c60a-b7bf-4a4b-8938-a8877edbe8ec",
                "content": "哇塞期待期待期待的去"
            }]
        })
    elif message['code'] == 6:
        # enter room
        room = enter_room(message)

        emit("message", {"code": 6,
                         "data": room}, boardcast=True)
        join_room(message['data']['rid'])

        rooms = get_rooms()

        emit('message', {
            "code": 4,
            "data": rooms
        }, broadcast=True)

        user_from = None
        users = get_users()
        for user in users:
            if user['id'] == message['data']['from']:
                user_from = user

        emit('message', {
            "code": 9,
            "data": {
                "rid": message['data']['rid'],
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from": '系统消息',
                "content": user_from['name'] + "加入了房间"
            }
        }, to=message['data']['rid'])

    elif message['code'] == 7:
        # leave room
        # print(message)
        room = l_room(message)
        # print('leave room')
        # print(room)
        # rooms = get_rooms()
        # emit("message", {"code": 7,
        #                  "data": room}, to=message['data']['rid'])
        emit("message", {"code": 7,
                         "data": room}, boardcast=True)

        # print(message['data']['id'])
        user_from = None
        users = get_users()
        for user in users:
            if user['id'] == message['data']['id']:
                user_from = user

        emit('message', {
            "code": 9,
            "data": {
                "rid": message['data']['rid'],
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from": '系统消息',
                "content": str(user_from['name']) + "退出了房间",
            }
        }, to=message['data']['rid'])

        leave_room(message['data']['rid'])
        rooms = get_rooms()
        emit('message', {
            "code": 4,
            "data": rooms
        }, broadcast=True)
    elif message['code'] == 8:
        del_room(message)
        emit('message', {
            "code": 8,
            "data": message['data']
        })
        leave_room(message['data'])

        rooms = get_rooms()
        # print(rooms)
        emit('message', {
            "code": 4,
            "data": rooms
        }, broadcast=True)


    elif message['code'] == 9:
        emit('message', {
            "code": 9,
            "data": {
                "rid": message['data']['rid'],
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from": message['data']['from'],
                "content": message['data']['content']
            }
        }, to=message['data']['rid'])
    elif message['code'] == 12:
        pass
        # emit('message', {"code": 11, "data": [
        #     {"id": "cf67d081-7637-4901-93af-9d4b3c824ba8", "name": "Tom", "status": "leisure"},
        #     {"id": "c6f2cf85-503e-4a97-b104-06e7964c08e1", "name": "unnamed", "status": "leisure"},
        #     {"id": "bcfccae8-5027-4a33-ae1b-11e789951eaa", "name": "unnamed", "status": "leisure"},
        #     {"id": "e54d5c3c-5618-447f-b255-eb0666f90450", "name": "unnamed", "status": "leisure"},
        #     {"id": "41c1faa2-d88f-4b96-a17f-fb6400672b39", "name": "unnamed", "status": "leisure"},
        #     {"id": "74d655af-3916-4400-9611-e8ad1618d5e4", "name": "unnamed", "status": "leisure"},
        #     {"id": "12b2b259-1812-4095-80c7-67420d146b67", "name": "unnamed", "status": "leisure"},
        #     {"id": "e5e3cc75-da05-4e89-9fce-4639a9821768", "name": "unnamed", "status": "leisure"},
        #     {"id": "f1572da9-8710-48d6-99c1-d8b85bc368df", "name": "unnamed", "status": "leisure"},
        #     {"id": "937cce34-3ecf-401e-953d-d85ae3d16960", "name": "unnamed", "status": "leisure"}]})

    elif message['code'] == 14:
        # print(message)
        room = set_ready(message)
        # print(room)
        emit('message', {'code': 14, 'data': room}, to=message['data']['rid'])
        room = get_room(message['data']['rid'])
        if (room['challenger']['name'] == 'random' or room['challenger']['name'] == 'expert' or room['challenger'][
            'name'] == 'Alpha bata' or room['challenger']['name'] == 'resnet' or room['challenger']['name'] == 'vit') and room['challenger']['ready'] == True and room['host'][
            'ready'] == True and room['challenger']['color'] == 0:
            game = GameState.new_game(5, 9)

            move_msg = ai_move(game, room)

            message = {'code': 15, 'data': move_msg}

            add_room_step(message)

            emit('message', message, to=message['data']['rid'])

            # send msg
            if len(move_msg['state']['move']) == 2:
                x_int = [int(i) + 1 for i in move_msg['state']['move']]
                send_str = "玩家 " + move_msg['name'] + " : " + "".join(str(num) for num in x_int)
            else:
                x, y = str(move_msg['state']['move'])[0:2], str(move_msg['state']['move'])[2:4]
                x_int = [int(i) + 1 for i in x]
                y_int = [int(i) + 1 for i in y]
                x = "".join(str(num) for num in x_int)
                y = "".join(str(num) for num in y_int)
                basic = "玩家 " + str(move_msg['name']) + " : " + x + ' ---> ' + y
                if move_msg['state']['eat'] != '-1' or move_msg['state']['eat'] != '' or move_msg['state']['eat'] != -1 or not move_msg['state']['eat'].startswith('-'):
                    eat_int = [int(i) + 1 for i in move_msg['state']['eat']]
                    append = ',吃子：' + "".join(str(num) for num in eat_int)
                    send_str = basic + append
                else:
                    send_str = basic

            emit('message', {
                "code": 9,
                "data": {
                    "rid": move_msg['rid'],
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "from": '系统消息',
                    "content": send_str
                }
            }, to=message['data']['rid'])
            # end


    elif message['code'] == 15:
        # print(message)

        # {'code': 15, 'data': {'rid': 'room2162', 'state': {
        #    'board': [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1, 0, -1, 0], [0, -1, 1, -1, 0],
        #              [0, -1, -1, -1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'move': '2242',
        #    'eat': '32', 'eat_c': 1,'player': 1, 'winner': 0, 'play_out': 1}, 'name': 7}}

        add_room_step(message)

        room = get_room(message['data']['rid'])
        if room['challenger']['name'] == 'random' or room['challenger']['name'] == 'expert' or room['challenger'][
            'name'] == 'Alpha beta' or room['challenger']['name'] == 'resnet' or room['challenger']['name'] == 'vit':

            if len(message['data']['state']['move']) == 2:
                x_int = [int(i) + 1 for i in message['data']['state']['move']]
                send_str = "玩家 " + message['data']['name'] + " : " + "".join(str(i) for i in x_int)
            else:
                x, y = str(message['data']['state']['move'])[0:2], str(message['data']['state']['move'])[2:4]
                x_int = [int(i) + 1 for i in x]
                y_int = [int(i) + 1 for i in y]
                x = "".join(str(num) for num in x_int)
                y = "".join(str(num) for num in y_int)
                basic = "玩家 " + str(message['data']['name']) + " : " + x + ' ---> ' + y
                # print(message['data']['state']['eat'])
                if message['data']['state']['eat'] != '-1' and message['data']['state']['eat'] != '' and \
                        message['data']['state']['eat'] != -1:
                    eat_int = [int(i) + 1 for i in message['data']['state']['eat']]
                    append = ',吃子：' + "".join(str(num) for num in eat_int)
                    send_str = basic + append
                else:
                    send_str = basic

            emit('message', {
                "code": 9,
                "data": {
                    "rid": message['data']['rid'],
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "from": '系统消息',
                    "content": send_str
                }
            }, to=message['data']['rid'])
            emit('message', {'code': 15, 'data': message['data']}, to=message['data']['rid'])

            game = game_model((message['data']['state']), room['steps'])

            move_msg = ai_move(game, room)
            add_room_step({'data': move_msg})
            emit('message', {'code': 15, 'data': move_msg}, to=message['data']['rid'])

            # send msg
            if len(move_msg['state']['move']) == 2:
                x_int = [int(i) + 1 for i in move_msg['state']['move']]
                send_str = "玩家 " + move_msg['name'] + " : " + "".join(str(num) for num in x_int)
            else:
                x, y = str(move_msg['state']['move'])[0:2], str(move_msg['state']['move'])[2:4]
                x_int = [int(i) + 1 for i in x]
                y_int = [int(i) + 1 for i in y]
                x = "".join(str(num) for num in x_int)
                y = "".join(str(num) for num in y_int)
                basic = "玩家 " + str(move_msg['name']) + " : " + x + ' ---> ' + y
                if move_msg['state']['eat'] != '-1' and move_msg['state']['eat'] != '' and move_msg['state'][
                    'eat'] != -1:
                    eat_int = [int(i) + 1 for i in move_msg['state']['eat']]
                    append = ',吃子：' + "".join(str(num) for num in eat_int)
                    send_str = basic + append
                else:
                    send_str = basic

            emit('message', {
                "code": 9,
                "data": {
                    "rid": move_msg['rid'],
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "from": '系统消息',
                    "content": send_str
                }
            }, to=message['data']['rid'])
            # end


        else:
            emit('message', {'code': 15, 'data': message['data']}, to=message['data']['rid'])
            if len(message['data']['state']['move']) == 2:
                send_str = "玩家 " + message['data']['name'] + " : " + str(message['data']['state']['move'])
            else:
                x, y = str(message['data']['state']['move'])[0:2], str(message['data']['state']['move'])[2:4]
                basic = "玩家 " + str(message['data']['name']) + " : " + x + ' ---> ' + y
                if message['data']['state']['eat'] != '-1' and message['data']['state']['eat'] != '':
                    eat_int = [int(i) + 1 for i in message['data']['state']['eat']]
                    append = ',吃子：' + ''.join(str(num) for num in eat_int)
                    send_str = basic + append
                    print(send_str)
                else:
                    send_str = basic

            emit('message', {
                "code": 9,
                "data": {
                    "rid": message['data']['rid'],
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "from": '系统消息',
                    "content": send_str
                }
            }, to=message['data']['rid'])
    elif message['code'] == 19:
        # game over
        # 修改数据，存数据
        # {'code': 19, 'data': {'rid': 'room6765', 'matchDetails': {'roomId': 'room6765', 'host': {'id': 8, 'name': 'root', 'status': 'chessing', 'role': 'host', 'color': 0, 'turn': False, 'ready': True}, 'challenger': {'id': 7, 'name': '123', 'status': 'chessing', 'role': 'challenger', 'color': 1, 'ready': True}}, 'color': 'black', 'cause': 'eat_11'}}
        # 如果黑吃11胜利：
        # {'code': 19, 'data': {'rid': 'rooma95ef381-ad40-41fa-a54a-7bfabb2ce610',
        #                       'matchDetails': {'roomId': 'rooma95ef381-ad40-41fa-a54a-7bfabb2ce610',
        #                                        'host': {'id': 5, 'name': '123', 'status': 'chessing', 'role': 'host',
        #                                                 'color': 0, 'turn': False, 'ready': True},
        #                                        'challenger': {'id': 6, 'name': 'random', 'status': 'chessing',
        #                                                       'role': 'challenger', 'color': 1, 'ready': True},
        #                                        'black_end_state': {
        #                                            'board': [[0, 0, -1, 0, 0], [0, -1, 0, 0, 0], [0, -1, -1, -1, -1],
        #                                                      [0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0],
        #                                                      [0, 0, -1, 0, 0], [0, -1, 0, -1, 0], [0, 0, -1, 0, 0]],
        #                                            'move': '5351', 'eat': '52', 'player': 1, 'winner': 1,
        #                                            'play_out': 27, 'eat_c': 11}}, 'color': 'black', 'cause': 'eat_11'}}

        # print(message)

        room = get_room(message['data']['rid'])
        if message['data']['cause'] == 'eat_11':
            room['steps'] += message['data']['matchDetails']['black_end_state']['move']

        if message['data']['color'] == 'black':
            if room['host']['color'] == 0:
                win = room['host']['id']
                win_name = room['host']['name']
                lose = room['challenger']['id']
                lose_name = room['challenger']['name']
                win_color = 1
            else:
                win = room['challenger']['id']
                win_name = room['challenger']['name']
                lose = room['host']['id']
                lose_name = room['host']['name']
                win_color = -1
        elif message['data']['color'] == 'white':
            if room['host']['color'] == 1:
                win = room['host']['id']
                win_name = room['host']['name']
                lose = room['challenger']['id']
                lose_name = room['challenger']['name']
                win_color = -1
            else:
                win = room['challenger']['id']
                win_name = room['challenger']['name']
                lose = room['host']['id']
                lose_name = room['host']['name']
                win_color = 1

        emit('message', {'code': 19, 'data': {'rid': message['data']['rid'], 'winner': {'name': win_name},
                                              'loser': {'name': lose_name}, 'cause': message['data']['cause']}},
             to=message['data']['rid'])
        #

        sgf = sgfModel(host=room['host']['id'], challenger=room['challenger']['id'], host_color=room['host']['color'],
                       winner=win, sgf=room['steps'])
        q_sgf = sgfModel.query.filter_by(sgf=room['steps']).first()
        if not q_sgf:
            result_win = winRateModel.query.filter_by(user_id=win).first()
            result_lose = winRateModel.query.filter_by(user_id=lose).first()
            db.session.add(sgf)
            if win_color == 1:
                if result_win is not None:
                    result_win.black_win += 1
                    db.session.add(result_win)
                if result_lose is not None:
                    result_lose.white_lose += 1
                    db.session.add(result_lose)
                else:
                    if result_win is None:
                        win_model = winRateModel(black_win=1, white_win=0, black_lose=0, white_lose=0, user_id=win)
                        db.session.add(win_model)
                    if result_lose is None:
                        lose_model = winRateModel(black_win=0, white_win=0, black_lose=0, white_lose=1, user_id=lose)
                        db.session.add(lose_model)

            if win_color == -1:

                if result_win is not None:
                    result_win.white_win += 1
                    db.session.add(result_win)
                if result_lose is not None:
                    result_lose.black_lose += 1
                    db.session.add(result_lose)
                else:
                    if result_win is None:
                        win_model = winRateModel(black_win=0, white_win=1, black_lose=0, white_lose=0, user_id=win)
                        db.session.add(win_model)
                    if result_lose is None:
                        lose_model = winRateModel(black_win=0, white_win=0, black_lose=1, white_lose=1, user_id=lose)
                        db.session.add(lose_model)

            db.session.commit()

        # if message['data']['color'] == 'white':
        #     emit('message', {'code': 15, 'data': {'rid': message['data']['rid'], 'winner': {
        #         'name': room['host']['name'] if room['host']['color'] == 1 else room['challenger']['color']}, 'loser': {
        #         'name': room['host']['name'] if room['host']['color'] == 0 else room['challenger']['color']},
        #                                           'cause': message['data']['cause']}}, to=message['data']['rid'])
        #     #
        #     sgf = sgfModel(host=room['host']['id'], challenger=room['challenger']['id'], sgf=room['steps'])
        #     db.session.add(sgf)
        #     db.session.commit()

    elif message['code'] == 16:
        # 撤销
        # print(message)

        room = get_room(message['data']['rid'])

        if message['data']['consent'] == 1:
            if len(room['steps']) >= 8:

                if room['challenger']['name'] == "random" or room['challenger']['name'] == 'expert' or \
                        room['challenger']['name'] == 'Alpha beta' or room['challenger']['name'] == 'resnet' or room['challenger']['name'] == 'vit':

                    # 需要悔两步
                    remove_room_step(message['data']['rid'])
                    remove_room_step(message['data']['rid'])

                    room = get_room(message['data']['rid'])

                    move_msg = {}

                    state = room_steps_2_state(room['steps'])

                    move_msg['rid'] = room['id']
                    move_msg['state'] = state
                    move_msg['name'] = message['data']['id']

                    emit('message', {'code': 15, 'data': move_msg}, to=message['data']['rid'])

                    emit('message', {
                        "code": 9,
                        "data": {
                            "rid": message['data']['rid'],
                            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "from": '系统消息',
                            "content": str(room['host']['name'] if message['data']['id'] == room['host']['id'] else
                                           room['challenger']['name']) + "悔棋"
                        }
                    }, to=message['data']['rid'])

                    emit('message',
                         {'code': 16, 'data': {'rid': room['id'], 'id': room['challenger']['id'], 'consent': 2}},
                         to=message['data']['rid'])  # # 人
                else:
                    emit('message', {'code': 16, 'data': message['data']}, to=message['data']['rid'])


            else:
                emit('message', {
                    "code": 9,
                    "data": {
                        "rid": message['data']['rid'],
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "from": '系统消息',
                        "content": "两步之后才能悔棋"
                    }
                }, to=message['data']['rid'])

        if message['data']['consent'] == 2:

            if len(room['steps']) >= 8:

                # {'code': 16, 'data': {'rid': 'room6790', 'id': 8, 'consent': 1}}

                # if room['challenger']['name'] == "random" or room['challenger']['name'] == 'expert' or \
                #         room['challenger']['name'] == 'Alpha beta':
                #
                # 需要悔两步
                remove_room_step(message['data']['rid'])
                remove_room_step(message['data']['rid'])

                room = get_room(message['data']['rid'])

                move_msg = {}

                state = room_steps_2_state(room['steps'])

                move_msg['rid'] = room['id']
                move_msg['state'] = state
                move_msg['name'] = message['data']['id']

                emit('message', {'code': 15, 'data': move_msg}, to=message['data']['rid'])

                emit('message', {
                    "code": 9,
                    "data": {
                        "rid": message['data']['rid'],
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "from": '系统消息',
                        "content": str(room['host']['name'] if message['data']['id'] == room['host']['id'] else
                                       room['challenger']['name']) + "悔棋"
                    }
                }, to=message['data']['rid'])
            #
            #     emit('message',
            #          {'code': 16, 'data': {'rid': room['id'], 'id': room['challenger']['id'], 'consent': 2}},
            #          to=message['data']['rid'])
            # else:
            # emit('message', {'code': 16, 'data': message['data']}, to=message['data']['rid'])

            else:
                emit('message', {
                    "code": 9,
                    "data": {
                        "rid": message['data']['rid'],
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "from": '系统消息',
                        "content": "两步之后才能悔棋"
                    }
                }, to=message['data']['rid'])

                # emit('message', {'code': 16, 'data': {'rid': room['id'], 'id': room['challenger']['id'], 'consent': 0}},
                #      to=message['data']['rid'])

        # join_room(room['host']['id'])

        # emit('message', {'code': 16, 'data': message['data']}, to=message['data']['rid'])

    elif message['code'] == 17:
        # 投降
        # {'code': 17, 'data': 'room06be2618-b926-444e-ba95-bfc5bb45772a'}

        pass

        # emit('message', {'code': 17, 'data': message['data']}, to=message['data']['rid'])

        # emit('message', {'code': 17, 'data': message['data']}, to=message['data']['id'])

    elif message['code'] == 21:
        # key_expire = listen_for_expirations()
        emit('message', {
            "code": 21,
            "data": {
                "rid": [i for i in expired_keys],
                'uid': [i for i in expired_users],
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "content": 'room expire'
            }
        }, boardcast=True)
        expired_keys.clear()
        expired_users.clear()
    elif message['code'] == 23:
        # print(message)
        t = redis.ttl(message['data'])
        emit('message', {
            "code": 23,
            "data": {
                "rid": message['data'],
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from": '系统消息',
                "t": t,
                "content": "房间将在" + str(t) + "s之后自动退出,请准备",
            }
        }, to=message['data'])
    elif message['code'] == 22:
        t = redis.ttl(message['data'])
        # print(t)
        emit('message', {
            "code": 22,
            "data": {
                "id": message['data'],
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # "from": '系统消息',
                "t": t,
            }
        })
