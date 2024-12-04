from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from flask_redis import FlaskRedis

db = SQLAlchemy()


socketio = SocketIO(logger=True, engineio_logger=True)


redis = FlaskRedis()




# def set_expiring_key(key, value, expire_time):
#     r.set(key, value, ex=expire_time)

