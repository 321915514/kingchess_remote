from sqlalchemy import func


from httpfront.exts import db
from httpfront.models.models import UserModel, sgfModel, winRateModel


# from httpfront.app import app

def get_sgf_info_by_user_id(user_name):
    users = UserModel.query.filter(UserModel.username.ilike(f'%{user_name}%')).all()
    for user in users:
        # 获取该用户作为主机的棋谱
        hosted_games = sgfModel.query.filter(sgfModel.host == user.id).all()
        # 获取该用户作为挑战者的棋谱
        challenged_games = sgfModel.query.filter(sgfModel.challenger == user.id).all()
        all_games = hosted_games + challenged_games
        results = []
        for sgf in all_games:
            host_user = UserModel.query.get(sgf.host)
            challenger_user = UserModel.query.get(sgf.challenger)
            if host_user and challenger_user:
                results.append({
                    'id': sgf.id,
                    'host_username': host_user.username,
                    'challenger_username': challenger_user.username,
                    'sgf': sgf.sgf,
                    'host_color': "白" if sgf.host_color == 1 else "黑",
                    'time': sgf.time.strftime('%Y-%m-%d %H:%M:%S'),
                    "length": len(sgf.sgf) // 4,
                    "winner": UserModel.query.get(sgf.winner).username,
                })
        return results
    else:
        return []


def get_all_sgf():
    all_sgf = sgfModel.query.all()
    results = []
    for sgf in all_sgf:
        host_user = UserModel.query.get(sgf.host)
        challenger_user = UserModel.query.get(sgf.challenger)
        if host_user and challenger_user:
            results.append({
                'id': sgf.id,
                'host_username': host_user.username,
                'challenger_username': challenger_user.username,
                'sgf': sgf.sgf,
                'host_color': "白" if sgf.host_color == 1 else "黑",
                'time': sgf.time.strftime('%Y-%m-%d %H:%M:%S'),
                "length": len(sgf.sgf) // 4,
                "winner": UserModel.query.get(sgf.winner).username,
            })

    sorted_data = sorted(results, key=lambda x: x['time'], reverse=True)
    return sorted_data


def get_win_data():
    # 查询并计算胜率
    results = (
        db.session.query(
            winRateModel,
            UserModel.username
        )
        .join(UserModel, winRateModel.user_id == UserModel.id).limit(50)
        .all()
    )

    # 构建返回结果
    response = []
    for win_rate, username in results:

        total_play = (win_rate.black_lose + win_rate.white_lose +
                      win_rate.black_win + win_rate.white_win)
        win_count = win_rate.black_win + win_rate.white_win
        win_rate_percentage = (100 * win_count / total_play) if total_play > 0 else 0

        response.append({
            'name': username,
            'black_win': win_rate.black_win,
            'white_win': win_rate.white_win,
            'win': win_count,
            'total_play': total_play,
            'win_rate': str(win_rate_percentage)+"%",
        })

    # 按照胜率排序
    response.sort(key=lambda x: x['win_rate'], reverse=True)
    for rank, item in enumerate(response, start=1):
        item['rank'] = rank

    # results.append({
    #     'name': user.username,
    #     'black_win': win_rate.black_win,
    #     'white_win': win_rate.white_win,
    #     # 'black_lose': win_rate.black_lose,
    #     # 'while_lose': win_rate.white_lose,
    #     "win": win_rate.black_win + win_rate.white_win,
    #     "total_play": win_rate.black_lose + win_rate.white_lose + win_rate.black_win + win_rate.white_win,
    #     "win_rate": 100 * (win_rate.black_win + win_rate.white_win) / (
    #                 win_rate.black_lose + win_rate.white_lose + win_rate.black_win + win_rate.white_win),
    # })

    return response

# if __name__ == '__main__':
#     with app.app_context():
#         get_win_data()

