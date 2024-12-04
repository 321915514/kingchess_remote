"""add black win white win

Revision ID: 7494b05dc154
Revises: 6e6733036213
Create Date: 2024-10-27 12:33:28.416685

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '7494b05dc154'
down_revision = '6e6733036213'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('result', schema=None) as batch_op:
        batch_op.add_column(sa.Column('black_win', sa.Integer(), nullable=False))
        batch_op.add_column(sa.Column('white_win', sa.Integer(), nullable=False))
        batch_op.add_column(sa.Column('black_lose', sa.Integer(), nullable=False))
        batch_op.add_column(sa.Column('white_lose', sa.Integer(), nullable=False))
        batch_op.drop_column('win')
        batch_op.drop_column('lose')

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('result', schema=None) as batch_op:
        batch_op.add_column(sa.Column('lose', mysql.INTEGER(display_width=11), autoincrement=False, nullable=False))
        batch_op.add_column(sa.Column('win', mysql.INTEGER(display_width=11), autoincrement=False, nullable=False))
        batch_op.drop_column('white_lose')
        batch_op.drop_column('black_lose')
        batch_op.drop_column('white_win')
        batch_op.drop_column('black_win')

    # ### end Alembic commands ###
