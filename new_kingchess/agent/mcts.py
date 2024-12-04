import copy
import math
import random

from agent import base
from agent.expert_agent import Expert_agent
from fundamental.board_simple import GameState
from fundamental.coordinate import Player, Point
from agent.random_agent_test import Random_agent
from fundamental.utils import print_board, print_move_go, print_move


# def show_tree(node,indent='',max_depth=3):
#     if max_depth<0:
#         return
#     if node is None:
#         return
#     if node.parent is None:
#         print('%sroot'%indent)
#     else:
#         player = node.parent.game_state.next_player
#         move = node.move
#         print('%s%s %s %d %.3f' %(indent,fmt(player),fmt(move),node.num_rollouts,node.winning_frac(player)))
#     for child in  sorted(node.children,key=lambda n:n.num_rollouts,reverse=True):
#         show_tree(child,indent+'  ',max_depth-1)


class MCTSNode:
    def __init__(self, game_state: GameState, parent=None, move=None):
        '''
        define
        :param game_state:
        :param parent:
        :param move:
        '''
        assert game_state is not None
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0
        # self.is_over_stage1 = False
        # self.eat_chess = 0
        # self.sum_chess = self.game_state.sum_chess
        # self.left_chess = self.game_state.left_chess
        self.is_over = self.game_state.game_over()[0]
        self.unvisited_moves = self.game_state.legal_moves()

    def add_random_child(self):
        '''
        负责向树中添加新的子节点
        :return:
        '''
        index = random.randint(0, len(self.unvisited_moves) - 1)


        new_move = self.unvisited_moves.pop(index)  # 弹出对应元素
        # index return pop

        # count_black, count_white = self.game_state.count_chess()
        # if self.game_state.stage == 1:
        #     new_node_state = self.game_state.apply_move_stage1(new_move)
        # elif (
        #         self.count_black < self.game_state.board.num_rows or self.count_white < self.game_state.board.num_rows) and not self.is_over:
        #     # print('stage3')
        #     new_node_state = self.game_state.apply_move_stage3(new_move)
        # else:
        #     new_node_state = self.game_state.apply_move_stage2(new_move)

        new_node_state = self.game_state.apply_move(new_move)

        new_node = MCTSNode(new_node_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        '''
        检测当前是否还有合法动作未添加到树中
        :return:
        '''
        return len(self.unvisited_moves) > 0

    def is_terminal(self):
        '''
        是否到达终盘
        :return:
        '''

        return self.is_over

    def winning_frac(self, player):
        '''
        推演获胜的比率
        应该换一种方式计算用q加u值
        :param player:
        :return:
        '''
        return float(self.win_counts[player]) / float(self.num_rollouts)


class MCTSAgent(base.Agent):
    '''
    首先用当前的游戏状态作为 根节点来创建一棵新搜索树，接着反复生成新的推演。在本节的实现中，每一回合执行固定轮数 的推演。在其他实现中，也有按照固定运行时长的。
　　每一轮推演开始，先沿着搜索树往下遍历，直至找到一个可以添加子节点的节点（即任何还 留有尚未添加到树中的合法动作的棋局）为止。select_move 负责挑选可供继续搜索的最佳分 支，我们先暂时忽略它的具体实现，将在 4.5.2 节中详细介绍。
　　找到合适的节点后，调用 add_random_child 来选择一个后续动作，并将它添加到搜索树 中。此时 node 是一个新创建的 MCTSNode，它还没有包含任何推演。
　　现在我们可以从这个节点调用 simulate_random_game 并开始推演了。simulate_random_game 的实现与第 3 章中介绍的 bot_v_bot 示例相同。
　　最后需要为新创建的节点以及它所有的祖先节点更新获胜统计信息。
    '''

    def __init__(self, num_rounds, temperature):
        base.Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_child(self, node):
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_rollouts = math.log(1e-10 if total_rollouts == 0 or total_rollouts < 0 else total_rollouts)
        best_score = -1
        best_child = None
        for child in node.children:
            win_percentage = child.winning_frac(node.game_state.player)

            exploration_factor = math.sqrt(log_rollouts / child.num_rollouts)

            uct_score = win_percentage + self.temperature * exploration_factor

            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    def select_move(self, game_state):

        # game_state = GameState.new_game(5, 9)
        # game_state.board = game.board
        # game_state.player = game.player
        # game_state.left_chess = game.left_chess
        # game_state.sum_chess = game.sum_chess
        # # new_game.play_out = game.play_out
        # # new_game.status = game.status
        # game_state.move = game.move
        # print(game_state.sum_chess)
        # print(game_state.left_chess)
        root = MCTSNode(game_state)
        # 首先用当前的游戏状态作为 根节点来创建一棵新搜索树
        # print(game_state.sum_chess)
        # print(game_state.left_chess)
        for i in range(self.num_rounds):
            node = root

            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)
            if node.can_add_child():
                # print(game_state.sum_chess)
                # print(game_state.left_chess)
                node = node.add_random_child()
            #     print(game_state.sum_chess)
            #     print(game_state.left_chess)
            # print('simulate_game')
            # print(node.game_state.sum_chess)
            # print(node.game_state.left_chess)

            winner = MCTSAgent.simulate_random_game(node.game_state)  # 模拟

            # print(node.game_state.sum_chess)
            # print(node.game_state.left_chess)
            # print('simulate_game_end')

            # print(str(winner))

            while node is not None:
                node.record_win(winner)
                node = node.parent

        scored_moves = [
            (child.winning_frac(game_state.player), child.move, child.num_rollouts) for child in root.children
        ]
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:5]:
            if m.point_ is None:
                print('%s - %.3f (%d)' % (m.point, s, n))
            else:
                print('%s%s - %.3f (%d)' % (m.point, m.point_, s, n))
        best_move = None
        best_pct = -1.0
        for child in root.children:
            child_pct = child.winning_frac(game_state.player)
            if child_pct > best_pct:
                best_pct = child_pct
                best_move = child.move
        if best_move.point_ is None:
            print('select move %s with win pct %.3f' % (best_move.point, best_pct))
        else:
            print('select move %s%s with win pct %.3f' % (best_move.point, best_move.point_, best_pct))
        return best_move

    @staticmethod
    def simulate_random_game(game_state):
        # new_game = GameState.new_game(5, 9)
        # new_game.board = game.board
        # new_game.player = game.player
        # new_game.left_chess = game.left_chess

        # # new_game.play_out = game.play_out
        # # new_game.status = game.status
        # new_game.move = game.move
        # print(game_state.eat_chess())
        # print(game_state.board.get_grid())
        bots = {
            Player.black: Expert_agent(),
            Player.white: Expert_agent()
        }

        while True:
            # if game_state.eat_chess() >= 11:
            #     # print(game_state.left_chess)
            #     return Player.black
            # print(game_state.sum_chess)
            # print_board(game_state.board)

            end, winner = game_state.game_over()

            if end:
                return winner
                # break

            if game_state.player == Player.black:
                move = bots[Player.black].select_move(game_state)
                # if move is None:
                #     return Player.white
            else:
                move = bots[Player.white].select_move(game_state)
            game_state = game_state.apply_move(move)

            # print(game.sum_chess)
            # print(game.board.get_grid())


