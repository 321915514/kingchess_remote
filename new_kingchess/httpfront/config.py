# db config
host = '127.0.0.1'
mysql_port = 3306
redis_port = 6379
username = 'root'
password = 'root'
database = 'chess'

db_url = f"mysql+pymysql://{username}:{password}@{host}:{mysql_port}/{database}?charset=utf8mb4"

SQLALCHEMY_DATABASE_URI = db_url

SECRET_KEY = 'secret_key'
# name_space = '/ws'

REDIS_URL = f'redis://{host}:{redis_port}/0'

ROOM = {"id": '', "dialog": [], "steps": '', "started": '',
        "host": {"id": '', "name": '', "status": "", "role": "host", "color": '', "turn": False,
                 "ready": False}, "challenger": {"id": "", "name": "", "status": "", "role": "", "color": '', "ready": False}, "spectators": [], "time": 0}
TTL = 10