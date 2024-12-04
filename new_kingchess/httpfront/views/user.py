from flask import Blueprint, request, jsonify, session

from httpfront.exts import db
from httpfront.exts import redis

from httpfront.models.models import UserModel
from httpfront.services.userService import set_user_status, set_user

user_bp = Blueprint('user', __name__, url_prefix='/user')
from werkzeug.security import generate_password_hash, check_password_hash


@user_bp.route('/login', methods=['POST'])
def login():
    content = request.get_json()

    user = UserModel.query.filter_by(username=content['username']).first()

    # print(user)

    if not user:
        return jsonify({'code': 6000, 'message': '用户不存在，请注册'})
    # print(check_password_hash(user.password, content['password']))
    if user.password == content['password']:
        # session['user-id'] = user.id

        # 0 leisure
        # set_user_status(user.id, 'leisure')
        set_user(user)
        # redis.set(user.id, 'leisure')

        return jsonify({'code': 2000, 'message': '登录成功', 'id': user.id})
    else:
        return jsonify({'code': 4000, 'message': '登录失败，密码错误！！！'})


@user_bp.route('/logout', methods=['POST'])
def logout():
    # session.pop('user-id')
    # ttl = redis.ttl(user.id)
    # if ttl >= 0:
    #     r.delete(key)
    return jsonify({'code': 2000, 'message': '登出成功'})


@user_bp.route('/register', methods=['POST'])
def register():
    content = request.get_json()
    users = UserModel.query.all()
    for user_db in users:
        if user_db.username == content['username']:
            return jsonify({'code': 4000, 'message': '用户名已存在'})
    else:
        user = UserModel(username=content['username'], password=content['password'])

        db.session.add(user)

        db.session.commit()

        return jsonify({'code': 2000, 'message': '注册成功'})


def check_user_session():
    if 'user-id' in session:
        return True
    else:
        return False


@user_bp.route('/get_ttl_user', methods=['POST'])
def get_ttl_user():
    content = request.get_json()
    t = redis.ttl(content)
    return jsonify({'code': 2000, 'message': t})
