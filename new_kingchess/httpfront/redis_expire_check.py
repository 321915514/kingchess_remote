import datetime

import redis
from flask_socketio import emit

# 连接到 Redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)

# 开启过期通知
r.config_set('notify-keyspace-events', 'Ex')


def set_expiring_key(key, value, expire_time):
    r.set(key, value, ex=expire_time)


def listen_for_expirations():
    r.config_set('notify-keyspace-events', 'Ex')
    pubsub = r.pubsub()
    pubsub.psubscribe('__keyevent@0__:expired')
    print("Listening for expired keys...")

    for message in pubsub.listen():
        if message['type'] == 'pmessage':
            key_expire = message['data'].decode()
            if key_expire.startswith('room'):
               print(key_expire)

# 设置即将过期的键
# set_expiring_key('test_key', 'Hello Redis', 5)

# 启动过期监听
if __name__ == '__main__':
    listen_for_expirations()
