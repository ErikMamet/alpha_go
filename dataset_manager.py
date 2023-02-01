import torch
import numpy as np 
import os
import os.path as osp
import pickle
import torchvision.transforms as T
import time

import copy
from torch.utils.data import Dataset



'''
This dataset was sourced from https://www.eugeneweb.com/pipermail/computer-go/2015-December/008353.html on 03/01/2023.
It has been pre-processed by (see utils.py)
This Dataset class works on a dataset structured in the following way :

- Dataset_dir
    - game_1
        - state_1  
        - state_2
        - ..
        - state_nb
        
    - game_2
    - ...
    - game_nb

where each state_nb contains a pickle file containing [Board, COLOR_VICTORY, next_move ,NB_MOVES_LEFT]
'''

def calc_nb_state_games(dataset_dir, nb_games):
    res = []
    for i in range(nb_games):
        res.append(len(os.listdir(osp.join(dataset_dir,"game_"+str(i)))))
    return res

def extract_boards(board):

    board_w, board_b = board.copy(), board.copy()
    board_w[board == 1] = 0
    board_b[board == 2] = 0
    return board_w, board_b

transf = T.ToTensor()

class Go9x9_Dataset(Dataset):
    def __init__(self, data_dir, size_of_input= 5, transform=transf, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        #self.nb_games = len(os.listdir(data_dir))
        self.nb_games = 9500
        self.nb_states_games = calc_nb_state_games(data_dir, self.nb_games)
        self.size_of_input = size_of_input

    def calc_game_board_id(self, idx):
        count = 0
        game_id = -1
        while count <= idx :
            game_id += 1
            count += self.nb_states_games[game_id]
        move_id = idx - count + self.nb_states_games[game_id]
        return game_id, move_id

    def __len__(self):
        return sum(self.nb_states_games)

    def __getitem__(self, idx):
        #print("START IMPORT ITEM")
        tt = time.time()
        game_id, board_id = self.calc_game_board_id(idx)
        board_path = osp.join(self.data_dir, "game_"+str(game_id), "state_"+str(board_id))
        t = time.time()
        with open(board_path, 'rb') as handle:
            Board, winner_color, last_player_color, next_move, number_moves_left = pickle.load(handle)
            Board = np.reshape(Board, (9,9))


        #print("STEP1", time.time() - t)
        if last_player_color == 1:
            fused_board = np.ones((1,9,9))
        if last_player_color == -1:
            fused_board = np.zeros((1,9,9))
        t = time.time()
        fused_boardb = 15*np.ones((1,9,9))
        fused_boardw = 15*np.ones((1,9,9))

        for i in range(self.size_of_input):
            if (board_id - i > 0):
                board_path = osp.join(self.data_dir, "game_"+str(game_id), "state_"+str(board_id-i))
                with open(board_path, 'rb') as handle:
                    Old_Board, winner_color, last_player_color, next_move, number_moves_left = pickle.load(handle)
                    assert type(Old_Board) == np.ndarray
                    board_w, board_b = extract_boards(Old_Board)
            
                    board_w, board_b = np.reshape(board_w, (1,9,9)), np.reshape(board_b, (1,9,9))
                
                if fused_boardb[0][0][0] == 15 : 
                    fused_boardw = board_w
                    fused_boardb = board_b
                else:
                    fused_boardw = np.vstack(( fused_boardw, board_w))
                    fused_boardb = np.vstack(( fused_boardb, board_b))

            else :
                if fused_boardb[0][0][0]  == 15 : 

                    fused_boardw = np.zeros((1,9,9))
                    fused_boardb = np.zeros((1,9,9))
                else :
                    fused_boardw = np.vstack(( fused_boardw, np.zeros((1,9,9)) ))
                    fused_boardb = np.vstack(( fused_boardb, np.zeros((1,9,9)) ))
        
        fused_board = np.vstack((fused_board, fused_boardw))
        fused_board = np.vstack((fused_board, fused_boardb))

        #print("STEP3", time.time() - t)

        one_hot = np.zeros(82)
        one_hot[next_move] = 1
        policy = one_hot
        if self.transform:
            fused_board = np.swapaxes(fused_board, 0, 1 )
            fused_board = np.swapaxes(fused_board, 1, 2 )
            fused_board = self.transform(fused_board)
        #label = [policy, winner_color*np.exp((1-number_moves_left)/9)]
        label = [policy, winner_color]

        return fused_board.type(torch.FloatTensor) , label


if __name__ == "__main__":
    DATASET_PATH = "./data/moyen_dataset"
    D = Go9x9_Dataset(DATASET_PATH)
    print(D.__getitem__(10))



    