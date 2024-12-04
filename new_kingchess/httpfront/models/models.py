from httpfront.exts import db
from datetime import datetime


class UserModel(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, auto_increment=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    join_time = db.Column(db.DateTime, default=datetime.now)
    hosted_games = db.relationship('sgfModel', foreign_keys='sgfModel.host', backref='host_user', lazy=True)
    challenged_games = db.relationship('sgfModel', foreign_keys='sgfModel.challenger', backref='challenger_user', lazy=True)
    win_rates = db.relationship('winRateModel', backref='user', lazy=True)

class sgfModel(db.Model):
    __tablename__ = "sgf"
    id = db.Column(db.Integer, primary_key=True, auto_increment=True)
    host = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    host_color = db.Column(db.Integer, nullable=False)
    challenger = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sgf = db.Column(db.String(500), nullable=False)
    winner = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    time = db.Column(db.DateTime, default=datetime.now)


class winRateModel(db.Model):
    __tablename__ = "result"
    id = db.Column(db.Integer, primary_key=True, auto_increment=True)
    black_win = db.Column(db.Integer, nullable=False)
    white_win = db.Column(db.Integer, nullable=False)
    black_lose = db.Column(db.Integer, nullable=False)
    white_lose = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
