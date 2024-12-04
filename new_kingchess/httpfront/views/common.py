
from flask import Blueprint, request, jsonify

from httpfront.services.commonService import get_all_sgf, get_win_data, get_sgf_info_by_user_id

common_bp = Blueprint('common', __name__, url_prefix='/common')


@common_bp.route('/get_all_sgf', methods=['POST', 'GET'])
def get_all_sgf_data():
    result = get_all_sgf()
    return jsonify({'code':2000,"message":result})

@common_bp.route('/get_sgf_by_user', methods=['POST', 'GET'])
def get_sgf_by_user_data():
    content = request.get_json()
    result = get_sgf_info_by_user_id(content)
    return jsonify({'code':2000,"message":result})


@common_bp.route('/get_win_rate', methods=['POST', 'GET'])
def get_win_rate_data():
    result = get_win_data()
    return jsonify({'code':2000,"message":result})


