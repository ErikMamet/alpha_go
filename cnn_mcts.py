import random
import math
import copy
import Goban
import numpy as np
import time
from go_cnn import GoCNN
import torch

'''
QUESTION:
- On arrête que grâce au max depth ?
- hasard sur égalité pour le choix des mouvements pour éviter d'avoir tendance à prendre le premier ? --> pas très utile
'''

class Node:

    def __init__(self, parent, color: int, move = []):
        self.parent = parent
        self.children = []
        self.visit = 0
        self.score = 0
        self.is_terminal = False
        self.terminal_score = 0
        self.color = color #0 if Black, 1 if White
        self.moves = copy.deepcopy(move)
        self.probas = None
        
    def expand(self, board:Goban.Board, model:GoCNN):
        '''Création de chaque enfant avec le rollout et backpropagation associée'''
        #On push les moves actuel du noeud
        for move in self.moves:
            board.push(move)
        #On boucle sur les nouveaux moves possibles
        for move in board.weak_legal_moves():
            child = Node(self, self.color ^ 1, self.moves) #On inverse la couleur
            child.moves.append(move)
            valid = board.push(move)
            if not valid:
                board.pop()
                continue
            self.children.append(child)
            child.is_terminal = board.is_game_over()
            if (child.is_terminal):
               result = board.result()
               score = self.get_result_score(result, child.color)
               child.terminal_score = score
               child.increase_score(score)
            board.pop()
        for _ in range(len(self.moves)):
            board.pop()
        #Select best child
        best_child = self.select_child(model)
        score = best_child.rollout(model)
        best_child.increase_score(score)


    def rollout(self, model:GoCNN):
        inputs = []
        board = Goban.Board()
        missing_boards = 6 - len(self.moves)
        if missing_boards > 0:
            for _ in range(missing_boards):
                inputs.append(copy.deepcopy(np.reshape(board._board, (9,9))))
        idx = len(self.moves)
        while len(inputs) < 6:
            for i in range(idx):
                board.push(self.moves[i])
            idx -= 1
            inputs.append(copy.deepcopy(np.reshape(board._board, (9,9))))
            board = Goban.Board()
        inputs = np.reshape(inputs,(1,6,9,9))
        inputs = torch.FloatTensor(inputs)
        policy, value = model(inputs)
        #best_move = np.argmax(policy.detach().cpu().numpy()[0])
        self.probas = policy.detach().cpu().numpy()[0]
        if (value.item() < 0): winner = -1
        elif (value.item() > 0): winner = 1
        else: winner = 0
        score = self.get_result_score(winner, self.color)
        return score

    def get_result_score(self, result, color):
        if result == -1: winner = 1 #black wins
        elif result == 1: winner = 2 #white wins
        else: winner = 0 #Deuce

        #On passe de 0 et 1 à 1 et 2 avec le +1
        if (winner == color+1): score = 1 #Victoire
        elif (winner == (color^1)+1): score = 0 #Defaite
        else: score = 0.5 #Deuce
        return(score)
        

    def __score__(self, child, proba) -> float:
        param = 4 * proba * math.sqrt((math.log(self.visit)/(1+child.visit)))
        #param = math.sqrt(2) * proba * math.sqrt((math.log(self.visit)/(1+child.visit)))
        score = (child.score / (1+child.visit)) + param
        return score

    def increase_score(self, score):
        self.score += score
        self.visit += 1
        actual_node = self
        while (actual_node.parent != None):
            actual_node.parent.score += score
            actual_node.parent.visit += 1
            actual_node = actual_node.parent

    def select_child(self, model):
        '''Sélection du meilleur enfant selon le compromis exploration/exploitation
        https://fr.wikipedia.org/wiki/Recherche_arborescente_Monte-Carlo
        '''
        if (self.probas is None): 
            score = self.rollout(model)
            self.increase_score(score)
        best_node = None
        best_score = -1
        for i in range(len(self.children)):
            #if self.children[i].is_terminal == False:
            score = self.__score__(self.children[i], self.probas[self.children[i].moves[-1]])
            if score > best_score:
                best_score = score
                best_node = self.children[i]
        return best_node

class MCTS:

    def __init__(self, root_node: Node, model: GoCNN, max_depth = 10, max_time = 300):
        self.root = root_node
        self.max_depth = max_depth
        self.board = Goban.Board()
        self.start_time = time.time()
        self.elapsed_time = 0
        self.max_time = max_time
        self.model = model
        score = self.root.rollout(self.model)
        self.root.increase_score(score)

    def play(self):
        '''Boucle de jeu'''
        self.actual_node = self.root
        depth = 0
        while (depth < self.max_depth and self.elapsed_time < self.max_time):
            print("new iteration, elapsed time is :", self.elapsed_time)
            while (self.actual_node.children != []):
                self.actual_node = self.actual_node.select_child(self.model)
            if (self.actual_node.is_terminal == True):
                print("pass")
                self.actual_node.increase_score(self.actual_node.terminal_score)
            else:
                self.actual_node.expand(self.board, self.model)
            depth = depth+1
            self.actual_node = self.root
            self.elapsed_time = time.time() - self.start_time
        print("end, elapsed time :", self.elapsed_time, "itération :", depth)


    def best_node(self):
        '''On retourne l'enfant de la racine qui a été le plus visité.'''
        max_visit = 0
        node_to_return = None
        for node in self.root.children:
            if node.visit > max_visit:
                node_to_return = node
                max_visit = node.visit
        return node_to_return

if __name__ == '__main__':
    model = GoCNN(6)
    model.load_state_dict(torch.load("log/10000_checkpoint_epoch_50"))

    root = Node(None, -1, [])
    mcts = MCTS(root, model, 10000, 30)
    mcts.play()
    for child in root.children:
        print(child.moves[-1])
        print(Goban.Board.flat_to_name(child.moves[-1]), " - ", child.score, "/", child.visit)
    for i in range(len(root.children)):
            #if self.children[i].is_terminal == False:
        score = root.__score__(root.children[i], root.probas[root.children[i].moves[-1]])
        print(Goban.Board.flat_to_name(root.children[i].moves[-1]), " - ", score, " - proba :", root.probas[root.children[i].moves[-1]])
    best_node = mcts.best_node()
    print(Goban.Board.flat_to_name(best_node.moves[-1]), " - ", best_node.score, "/", best_node.visit)

