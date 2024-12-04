# -*- coding: utf-8 -*-
import random

import numpy as np
import copy

import torch

from agent.expert_agent import Expert_agent
from fundamental.board import GameState, predict_reward
from fundamental.utils import print_board


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        '''
        :param parent: 父节点
        :param prior_p: 先验概率
        '''
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0  # 搜索过程中一直跟踪
        self._u = 0
        self._P = prior_p  # 先验概率 由神经网络给出

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # if full_search:
        for action, prob in action_priors:
            # if action not in self._children:
            self._children[action] = TreeNode(self, prob)
        # else:
        # for i in range(len(action_priors)):
        #     if mcts.node_num < 300:
        #         # if action not in self._children:
        #         if action_priors[i] > 0:
        #             self._children[i] = TreeNode(self, action_priors[i], i)
        #             mcts.count_node()
        #         else:
        #             break

    def select(self, state, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(state, c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, state:GameState, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """

        # expert_knowledge_value = predict_reward(state, self.action)
        # scores = Expert_agent().score_moves(state)
        # min_value = min(scores.values())
        # max_value = max(scores.values())

        # 归一化字典中的值
        # normalized_expert_knowledge = {k: (v - min_value) / (max_value - min_value+1e-10) for k, v in
        #                                scores.items()}
        # + normalized_expert_knowledge[state.a_trans_move(self.action)])
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.node_num = 0

    def _playout(self, state: GameState):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break
            # Greedily select next move.

            index, node = node.select(state, self._c_puct)
            # print(state.a_trans_move(index))
            state = state.apply_move(state.a_trans_move(index))
            # print_board(state.board)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        #print("current player: {},vlaue:{}".format(state.player, leaf_value))

        end, winner = state.game_over()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            # if winner == -1:  # tie
            #     leaf_value = 0.0
            # else:
            leaf_value = (1.0 if winner == state.player else -1.0)

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def count_node(self):
        self._node_num += 1

    def get_move_probs(self, state: GameState, mcts, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        self._node_num = 0
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node

        act_visits = [(action, node._n_visits) for action, node in self._root._children.items()]

        # print(act_visits)

        acts, visits = zip(*act_visits)
        
        #print(visits)

        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)


    def select_move(self, game: GameState, temp=1e-3, return_prob=0):
        # sensible_moves = game.legal_moves()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        # move_probs = np.zeros(len(sensible_moves))
        # move_probs = {}
        # if len(sensible_moves) > 0:
        # _, pos_moves = game.legal_position()
        move_probs = np.zeros([1125])
        acts, probs = self.mcts.get_move_probs(game, self.mcts, temp)
        #print(probs)
        probs = 0.75 * probs + 0.25 * np.random.dirichlet(0.6 * np.ones(len(probs)))

        probs /= probs.sum()

        move_probs[list(acts)] = probs
        # # acts (move,move,move,...) probs = [0.11111111 0.11111111 0.         0.         0.11111111 0.11111111,
        # 0.11111111 0.         0.11111111 0.         0.11111111 0., 0.11111111 0.         0.         0.11111111]

        if self._is_selfplay:
            # add Dirichlet Noise for exploration (needed for
            # self-play training)
            move = np.random.choice(
                acts,
                p=probs
            )
            # update the root node and reuse the search tree
            self.mcts.update_with_move(move)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob

            move_index = np.argmax([i for i in probs])

            # move = np.random.choice(acts, p=probs)   #   2024/4/5 modify
            move = acts[move_index]
            # reset the root node
            self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))
            # pass
        if return_prob:
            return move, move_probs
        else:
            return move


    def get_action(self, game: GameState, temp=1e-3, return_prob=0):
        # sensible_moves = game.legal_moves()
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        # move_probs = np.zeros(len(sensible_moves))
        # move_probs = {}
        # if len(sensible_moves) > 0:
        # _, pos_moves = game.legal_position()
        move_probs = np.zeros([1125])
        acts, probs = self.mcts.get_move_probs(game, self.mcts, temp)
        probs = 0.75 * probs + 0.25 * np.random.dirichlet(0.6 * np.ones(len(probs)))

        probs /= probs.sum()

        move_probs[list(acts)] = probs
        # # acts (move,move,move,...) probs = [0.11111111 0.11111111 0.         0.         0.11111111 0.11111111,
        # 0.11111111 0.         0.11111111 0.         0.11111111 0., 0.11111111 0.         0.         0.11111111]

        if self._is_selfplay:
            # add Dirichlet Noise for exploration (needed for
            # self-play training)
            move = np.random.choice(
                acts,
                p=probs
            )
            # update the root node and reuse the search tree
            self.mcts.update_with_move(move)
        else:
            # with the default temp=1e-3, it is almost equivalent
            # to choosing the move with the highest prob

            # move_index = np.argmax([i for i in probs])

            move = np.random.choice(acts, p=probs)   #   2024/4/5 modify
            # move = acts[move_index]
            # reset the root node
            self.mcts.update_with_move(-1)
            #                location = board.move_to_location(move)
            #                print("AI move: %d,%d\n" % (location[0], location[1]))
            # pass
        if return_prob:
            return move, move_probs
        else:
            return move

    # def select_move(self, game: GameState, full_search=1, temp=1e-3, return_prob=0):
    #     # sensible_moves = game.legal_moves()
    #     # the pi vector returned by MCTS as in the alphaGo Zero paper
    #     # move_probs = np.zeros(len(sensible_moves))
    #     # move_probs = {}
    #     # if len(sensible_moves) > 0:
    #
    #     acts, probs = self.mcts.get_move_probs(game, full_search, self.mcts, temp)
    #
    #     # # acts (move,move,move,...) probs = [0.11111111 0.11111111 0.         0.         0.11111111 0.11111111,
    #     # 0.11111111 0.         0.11111111 0.         0.11111111 0., 0.11111111 0.         0.         0.11111111]
    #
    #     move_probs = list(zip(acts, probs))
    #     if self._is_selfplay:
    #         # add Dirichlet Noise for exploration (needed for
    #         # self-play training)
    #         move = np.random.choice(
    #             acts,
    #             p=0.75 * probs + 0.25 * np.random.dirichlet(0.6 * np.ones(len(probs)))
    #         )
    #         # update the root node and reuse the search tree
    #         self.mcts.update_with_move(move)
    #     else:
    #         # with the default temp=1e-3, it is almost equivalent
    #         # to choosing the move with the highest prob
    #
    #         move_index = np.argmax([i[1] for i in move_probs])
    #
    #         move = move_probs[move_index][0]
    #
    #         # move = np.random.choice(acts, p=probs)   #   2024/4/5 modify
    #
    #         # reset the root node
    #         self.mcts.update_with_move(-1)
    #         #                location = board.move_to_location(move)
    #         #                print("AI move: %d,%d\n" % (location[0], location[1]))
    #         pass
    #     if return_prob:
    #         return move, move_probs
    #     else:
    #         return move

    def __str__(self):
        return "MCTS {}".format(self.player)


if __name__ == '__main__':
    # game = GameState.new_game(5, 9)
    # mcts_player = MCTSPlayer(PolicyValueNet(9, 9).policy_value_fn,
    #                               c_puct=5,
    #                               n_playout=10,
    #                               is_selfplay=1)
    #
    # mcts_player.get_action(game)
    pass
