import Goban
import numpy as np
import matplotlib.pyplot as plt
from cnn_mcts import Node, MCTS
from multi_cnn_mcts import Multi_MCTS
from playerInterface import *
from go_cnn import GoCNN
import torch


class myPlayer(PlayerInterface):
    ''' Abstract class that must be implemented by you AI. Typically, a file "myPlayer.py" will implement it for your
    AI to enter the tournament.
    
    You may want to check to player implementations of this interface:
    - the random player
    - the gnugo player
    '''

    def __init__(self):
        self._board = Goban.Board()
        self._mycolor = None
        self.move = []

    def getPlayerName(self):
        ''' Must return the name of your AI player.'''
        return "MCTS Player"


    def getPlayerMove(self): 
        ''' This is where you will put your AI. This function must return the move as a standard
        move in GO, ie, "A1", "A2", ..., "D5", ..., "J8", "J9" or "PASS"

        WARNING: In the Board class, legal_moves() and weak_legal_moves() are giving internal
        coordinates only (to speed up the push/pop methods and the game tree traversal). However,
        to communicate with this interface, you can't use these moves anymore here.

        You have to use the helper function flat_to_name to translate the internal representation of moves
        in the Goban.py file into a named move.

        The result of this function must be one element of [Board.flat_to_name(m) for m in b.legal_moves()]
        (it has to be legal, so at the end, weak_legal_moves() may not be sufficient here.)
        '''
        root = Node(None, self._mycolor-1, self.move)
        #mcts = MCTS(root, model, 1000, 30)
        multi_mcts = Multi_MCTS(root,10000,60)
        #mcts = MCTS(root,5)
        #mcts.play()
        #best_node = mcts.best_node()
        best_node = multi_mcts.main()
        self.move.append(best_node.moves[-1])
        return Goban.Board.flat_to_name(best_node.moves[-1]) 

    def playOpponentMove(self, move): 
        '''Inform you that the oponent has played this move. You must play it with no 
        search (just update your local variables to take it into account)

        The move is given as a GO move string, like "A1", ... "J9", "PASS"
         
        WARNING: because the method Goban.push(m) needs a move represented as a flat move (integers),
        you can not directly call this method with the given move here. You will typically call
        b.push(Board.name_to_flat(move)) to translate the move into its flat (internal) representation. 
         '''
        self.move.append(Goban.Board.name_to_flat(move))
        self._board.push(Goban.Board.name_to_flat(move))

    def newGame(self, color): 
        '''Starts a new game, and give you your color.  As defined in Goban.py : color=1
        for BLACK, and color=2 for WHITE'''
        self._mycolor = color
        self._opponent = Goban.Board.flip(color)

    def endGame(self, color):
        '''You can get a feedback on the winner
        This function gives you the color of the winner'''
        if self._mycolor == color:
            print("I won!!!")
        else:
            print("I lost :(!!")


