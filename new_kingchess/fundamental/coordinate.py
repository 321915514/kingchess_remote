import enum
from collections import namedtuple


class Player(enum.Enum):
    black = 1
    white = -1
    draw = 0

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white


class Move:
    def __init__(self, point=None, point_=None):
        assert (point is not None)
        self.point = point
        self.point_ = point_
        self.is_down = (self.point is not None and self.point_ is None)
        self.is_go = (self.point is not None and self.point_ is not None)

    @classmethod
    def play_down(cls, point):
        return Move(point=point)

    @classmethod
    def play_go(cls, point, point_):
        return Move(point=point, point_=point_)

    def __str__(self):
        if self.is_down:
            return str(self.point.row) + str(self.point.col)
        else:
            return str(self.point.row) + str(self.point.col) + str(self.point_.row) + str(self.point_.col)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Move):
            return self.point == other.point and self.point_ == other.point_
        return False

    def __hash__(self):
        return hash((self.point, self.point_))



# 指定落子的位置
class Point(namedtuple('Point', 'row col')):


    def point_2_index(self):
        return self.col + self.row * 9

    # def neighbors(self):
    #     return [
    #         Point(self.row + 1, self.col),  # 上
    #         Point(self.row + 1, self.col + 1),  # 右上
    #         Point(self.row, self.col + 1),  # 右
    #         Point(self.row - 1, self.col + 1),  # 右下
    #         Point(self.row - 1, self.col), # 下
    #         Point(self.row - 1, self.col-1),  # 下左
    #         Point(self.row, self.col - 1), # 左
    #         Point(self.row + 1, self.col - 1),  # 左上
    #
    #         Point(self.row + 2, self.col),  # 上上
    #         Point(self.row + 2, self.col + 2),  # 右上右下
    #         Point(self.row, self.col + 2),  # 右右
    #         Point(self.row - 2, self.col + 2),  # 右下右下
    #         Point(self.row - 2, self.col),  # 下下
    #         Point(self.row - 2, self.col - 2),  # 下左下左
    #         Point(self.row, self.col - 2),  # 左左
    #         Point(self.row + 2, self.col - 2),  # 左上左上
    #     ]

    def border_neighbors(self):
        coord = self.col + self.row * 9

        if coord == 0:
            return [
                Point(self.row+2, self.col),
                Point(self.row+1,self.col+1),

                [Point(self.row+2, self.col), Point(self.row+4,self.col)],
                [Point(self.row+1,self.col+1), Point(self.row+2, self.col+2)],
                2,
            ]
        if coord == 2:
            return [
                Point(self.row+1, self.col),
                Point(self.row+1,self.col+1),
                Point(self.row, self.col + 1),

                [Point(self.row+1, self.col), Point(self.row+2, self.col)],
                [Point(self.row+1,self.col+1), Point(self.row+2, self.col+2)],
                [Point(self.row, self.col + 1), Point(self.row, self.col + 2)],
                3,
            ]
        if coord == 8:
            return [
                Point(self.row+2, self.col),
                Point(self.row+1,self.col-1),

                [Point(self.row+2, self.col),Point(self.row+4,self.col)],
                [Point(self.row+1,self.col-1),Point(self.row+2,self.col-2)],
                2
            ]
        if coord == 10: ### 整体反转 modify successful
            return [
                Point(self.row + 1, self.col),
                Point(self.row + 1, self.col + 1),
                Point(self.row - 1, self.col - 1),

                [Point(self.row + 1, self.col), Point(self.row + 2, self.col)],
                [Point(self.row + 1, self.col + 1), Point(self.row + 2, self.col+2)],
                3
            ]
        if coord == 16: ### 整体反转modify successful
            return [
                Point(self.row + 1, self.col),
                Point(self.row - 1, self.col + 1),
                Point(self.row + 1, self.col - 1),

                [Point(self.row + 1, self.col), Point(self.row + 2, self.col)],
                [Point(self.row + 1, self.col - 1), Point(self.row + 2, self.col - 2)],
                3
            ]
        if coord == 18: ### 将第一个加到末尾 successful
            return [
                Point(self.row + 2, self.col),
                Point(self.row, self.col+1),
                Point(self.row - 2, self.col),

                [Point(self.row, self.col+1), Point(self.row, self.col+2)],
                3
            ]
        # 严格
        if coord == 19:
            return [
                Point(self.row + 1, self.col),
                Point(self.row, self.col + 1),
                Point(self.row - 1, self.col),
                Point(self.row, self.col-1),

                [Point(self.row, self.col + 1), Point(self.row, self.col + 2)],
                4
            ]
        if coord == 25:
            return [
                Point(self.row + 1, self.col),
                Point(self.row, self.col + 1),
                Point(self.row - 1, self.col),
                Point(self.row, self.col - 1),

                [Point(self.row, self.col - 1), Point(self.row, self.col - 2)],
                4
            ]
        if coord == 26: ### 前面两个交换successful
            return [
                Point(self.row + 2, self.col),
                Point(self.row - 2, self.col),
                Point(self.row, self.col - 1),

                [Point(self.row, self.col - 1), Point(self.row, self.col - 2)],
                3
            ]
        if coord == 28:
            return [
                Point(self.row - 1, self.col + 1),
                Point(self.row - 1, self.col),
                Point(self.row+1, self.col - 1),

                [Point(self.row - 1, self.col + 1), Point(self.row-2, self.col + 2)],
                [Point(self.row - 1, self.col), Point(self.row - 2, self.col)],
                3
            ]
        if coord == 29:  ### 整体反转successful
            return [
                Point(self.row + 1, self.col),
                Point(self.row, self.col + 1),
                Point(self.row - 1, self.col),

                [Point(self.row, self.col+1), Point(self.row, self.col + 2)],
                [Point(self.row - 1, self.col), Point(self.row - 2, self.col)],
                3
            ]
        if coord == 33:  ### 前两个反转successful
            return [
                Point(self.row + 1, self.col),
                Point(self.row - 1, self.col),
                Point(self.row, self.col-1),

                [Point(self.row - 1, self.col), Point(self.row - 2, self.col)],
                [Point(self.row, self.col-1), Point(self.row, self.col-2)],
                3
            ]
        if coord == 34:
            return [
                Point(self.row + 1, self.col+1),
                Point(self.row - 1, self.col),
                Point(self.row-1, self.col - 1),

                [Point(self.row - 1, self.col), Point(self.row - 2, self.col)],
                [Point(self.row-1, self.col - 1), Point(self.row-2, self.col - 2)],
                3
            ]
        if coord == 36:
            return [
                Point(self.row - 1, self.col+1),
                Point(self.row - 2, self.col),

                [Point(self.row - 1, self.col+1), Point(self.row-2, self.col +2)],
                [Point(self.row - 2, self.col), Point(self.row - 4, self.col)],
                2

            ]
        if coord == 38:
            return [
                Point(self.row, self.col + 1),
                Point(self.row - 1, self.col + 1),
                Point(self.row - 1, self.col),

                [Point(self.row, self.col + 1), Point(self.row, self.col + 2)],
                [Point(self.row - 1, self.col + 1), Point(self.row-2, self.col + 2)],
                [Point(self.row - 1, self.col), Point(self.row - 2, self.col)],
                3

            ]
        if coord == 42:
            return [
                Point(self.row-1, self.col),
                Point(self.row - 1, self.col - 1),
                Point(self.row, self.col-1),

                [Point(self.row-1, self.col), Point(self.row-2, self.col)],
                [Point(self.row - 1, self.col - 1), Point(self.row - 2, self.col - 2)],
                [Point(self.row, self.col-1), Point(self.row, self.col-2)],
                3


            ]
        if coord == 44:
            return [
                Point(self.row - 2, self.col),
                Point(self.row - 1, self.col-1),

                [Point(self.row - 2, self.col), Point(self.row - 4, self.col)],
                [Point(self.row - 1, self.col-1), Point(self.row - 2, self.col-2)],
                2
            ]
        else:
            return [
                Point(self.row + 1, self.col),  # 上
                Point(self.row + 1, self.col + 1),  # 右上
                Point(self.row, self.col + 1),  # 右
                Point(self.row - 1, self.col + 1),  # 右下
                Point(self.row - 1, self.col),  # 下
                Point(self.row - 1, self.col - 1),  # 下左
                Point(self.row, self.col - 1),  # 左
                Point(self.row + 1, self.col - 1),  # 左上

                [Point(self.row + 1, self.col), Point(self.row + 2, self.col)],  # 上上
                [Point(self.row + 1, self.col + 1), Point(self.row + 2, self.col + 2)],  # 右上右下
                [Point(self.row, self.col + 1), Point(self.row, self.col + 2)],  # 右右
                [Point(self.row - 1, self.col + 1), Point(self.row - 2, self.col + 2)],  # 右下右下
                [Point(self.row - 1, self.col), Point(self.row - 2, self.col)],  # 下下
                [Point(self.row - 1, self.col - 1), Point(self.row - 2, self.col - 2)],  # 下左下左
                [Point(self.row, self.col - 1), Point(self.row, self.col - 2)],  # 左左
                [Point(self.row + 1, self.col - 1), Point(self.row + 2, self.col - 2)],  # 左上左上
                8
            ]


    # def __str__(self):
    #     return 'Point(row=%s,col=%d)' %(self.row,self.col)



