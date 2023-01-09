import random
import math
import copy
import Goban
import numpy as np

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
        
    def expand(self, board:Goban.Board):
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
            else: score = child.rollout()
            child.increase_score(score)
            board.pop()
        for i in range(len(self.moves)):
            board.pop()

    def rollout(self):
        board = Goban.Board()
        for move in self.moves:
            board.push(move)
        while (board.is_game_over() == False):
            moves = board.weak_legal_moves()
            is_legal = False
            while(is_legal == False):
                probabilities = np.random.uniform(size=len(moves))
                valid = board.push(moves[np.argmax(probabilities)])
                if not valid:
                    board.pop()
                else: is_legal = True
        result = board.result()
        score = self.get_result_score(result, self.color)
        return score

    def get_result_score(self, result, color):
        if (result == "1-0"): winner = 2 #White wins
        elif (result == "0-1"): winner = 1 #Black wins
        else: winner = 0 #Deuce 

        #On passe de 0 et 1 à 1 et 2 avec le +1
        if (winner == color+1): score = 1 #Victoire
        elif (winner == (color^1)+1): score = 0 #Defaite
        else: score = 0.5 #Deuce
        return(score)
        

    def __score__(self, child) -> float:
        param = math.sqrt(2) * math.sqrt((math.log(self.visit)/child.visit))
        score = (child.score / child.visit) + param
        return score

    def increase_score(self, score):
        self.score += score
        self.visit += 1
        actual_node = self
        while (actual_node.parent != None):
            actual_node.parent.score += score
            actual_node.parent.visit += 1
            actual_node = actual_node.parent

class MCTS:
    def __init__(self, root_node: Node, max_depth = 10):
        self.root = root_node
        self.max_depth = max_depth
        self.board = Goban.Board()

    def play(self):
        '''Boucle de jeu'''
        self.actual_node = self.root
        depth = 0
        while (depth < self.max_depth):
            print("new iteration")
            while (self.actual_node.children != []):
                self.actual_node = self.select_child(self.actual_node)
            if (self.actual_node.is_terminal == True):
                print("pass")
                self.actual_node.increase_score(self.actual_node.terminal_score)
            else:
                self.actual_node.expand(self.board)
            depth = depth+1
            self.actual_node = self.root

            
    def select_child(self, node: Node):
        '''Sélection du meilleur enfant selon le compromis exploration/exploitation
        https://fr.wikipedia.org/wiki/Recherche_arborescente_Monte-Carlo
        '''
        best_node = None
        best_score = -1
        for child in node.children:
            if child.is_terminal == False:
                score = node.__score__(child)
                if score > best_score:
                    best_score = score
                    best_node = child
        return best_node

    def best_node(self):
        '''On retourne l'enfant de la racine qui a été le plus visité.'''
        max_visit = 0
        node_to_return = None
        for node in self.root.children:
            if node.visit > max_visit:
                node_to_return = node
                max_visit = node.visit
        return node_to_return


root = Node(None, 1)
mcts = MCTS(root,1)
mcts.play()
for child in root.children:
    print(child.moves[-1])
    print(Goban.Board.flat_to_name(child.moves[-1]), " - ", child.score, "/", child.visit)
best_node = mcts.best_node()
print(Goban.Board.flat_to_name(best_node.moves[-1]), " - ", best_node.score, "/", best_node.visit)

