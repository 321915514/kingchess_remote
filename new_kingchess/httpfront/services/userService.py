import json

from httpfront.exts import redis
from httpfront.models.models import UserModel

def get_user_status(id):
    if redis.get(id) is not None:
        return redis.get(id).decode('utf-8')
    else:
        return ''


def set_user_status(id, status):
    redis.set(id, status)

def set_user(user:UserModel):
    if len(get_users()) == 0:
        # redis.rpush('user', json.dumps({'id': user.id, 'name': user.username, 'status': 'leisure'}))
        redis.setex(user.id, 3*60*60, json.dumps({'id': user.id, 'name': user.username, 'status': 'leisure'}))
    else:
        user_set = set()
        for user_redis in get_users():
            user_set.add(user_redis['id'])
        if user.id in user_set:
            return
        else:
            redis.setex(user.id, 3*60*60, json.dumps({'id': user.id, 'name': user.username, 'status': 'leisure'}))
            # redis.rpush('user', json.dumps({'id':user.id,'name':user.username,'status':'leisure'}))


def get_user(uid):
    return json.loads(redis.get(uid).decode('utf-8'))


def get_users():
    result = []
    for i in redis.keys():
        if i.decode('utf-8').isdigit():
            result.append(get_user(i.decode('utf-8')))
    return result

    # return [eval(i.decode('utf-8')) for i in redis.lrange('user', 0, -1)]