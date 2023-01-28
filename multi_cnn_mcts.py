from multiprocess import Pool
from cnn_mcts import Node, MCTS
import numpy as np
import copy
from itertools import groupby
from go_cnn import GoCNN
import torch
import Goban

class Multi_MCTS:
    model = GoCNN(6)
    model.load_state_dict(torch.load("log/10000_checkpoint_epoch_50"))

    def __init__(self, root: Node, max_depth = 10, max_time = 300, nb_processes = 6) -> None:
        self.root = root
        self.max_time = max_time
        self.max_depth = max_depth
        self.nb_processes = nb_processes

    def run_mcts(self, i: int):
        #Debug l'inférence qui est ralentie à cause de OpenMP : 
        # https://stackoverflow.com/questions/65057388/pytorch-multiprocessing-with-shared-memory-causes-matmul-to-be-30x-slower-with
        torch.set_num_threads(1)
        #
        np.random.seed(i)
        root = copy.deepcopy(self.root)
        mcts = MCTS(root, self.model, self.max_depth, self.max_time)
        mcts.play()
        return root

    def combine_mcts(self, roots):
        all_children = [item for root in roots for item in root.children]
        res = [list(v) for l,v in groupby(sorted(all_children, key=lambda x:x.moves), lambda x: x.moves)]
        merged_children = []
        for child in res:
            merged_child = Node(self.root, child[0].color, child[0].moves)
            for item in child:
                merged_child.visit += item.visit
                merged_child.score += item.score
            merged_children.append(merged_child)
        self.root.children = merged_children

    def best_node(self, node: Node):
        '''On retourne l'enfant de la racine qui a été le plus visité.'''
        max_visit = 0
        node_to_return = None
        for child in node.children:
            if child.visit > max_visit:
                node_to_return = child
                max_visit = child.visit
        return node_to_return

    def main(self):
        seeds = [i for i in range(self.nb_processes)]
        with Pool(self.nb_processes) as p:
            results = p.map(self.run_mcts, seeds)
        self.combine_mcts(results)
        return self.best_node(self.root)

if __name__ == '__main__':
    root = Node(None, -1, [])
    multi_mcts = Multi_MCTS(root,1000,60)
    best_node = multi_mcts.main()