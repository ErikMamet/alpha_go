import torch
import numpy as np 
import os
import os.path as osp
import pickle
import torchvision.transforms as T

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



'''
This dataset was sourced from [insert link] on 03/01/2023. It has been pre-processed by (see [insert final pre process file name])
This Dataset class works on a datset structured in the following way :

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


transf = T.ToTensor()

class Go9x9_Dataset(Dataset):
    def __init__(self, data_dir, size_of_input= 5, transform=transf, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.nb_games = len(os.listdir(data_dir))
        self.nb_states_games = calc_nb_state_games(data_dir, self.nb_games)
        self.size_of_input = size_of_input

    def calc_game_board_id(self, idx):
        count = 0
        game_id = -1
        while count < idx :
            count += self.nb_states_games[game_id]
            game_id += 1
        move_id = idx - count + self.nb_states_games[game_id]
        return game_id, move_id



    def __len__(self):
        return sum(self.nb_states_games)

    def __getitem__(self, idx):
        
        game_id, board_id = self.calc_game_board_id(idx)
        board_path = osp.join(self.data_dir, "game_"+str(game_id), "state_"+str(board_id))
        with open(board_path, 'rb') as handle:
            Board, winner_color, last_player_color, next_move, number_moves_left = pickle.load(handle)
            Board = np.reshape(Board, (9,9))

        if last_player_color == -1:
            fused_board = np.zeros((9,9), dtype='int8')
        if last_player_color == 1:

            fused_board = np.ones((9,9), dtype='int8')
        
        for i in range(self.size_of_input):
            if i<self.nb_states_games[game_id]:
                board_path = osp.join(self.data_dir, "game_"+str(game_id), "state_"+str(board_id-i))
                with open(board_path, 'rb') as handle:
                    Old_Board, winner_color, last_player_color, next_move, number_moves_left = pickle.load(handle)
                    Old_Board = np.reshape(Old_Board, (9,9))
                
                fused_board = np.stack(( fused_board, Old_Board))
            else :
                fused_board = np.stack(( fused_board, np.zeros((9,9)) ))


        enocoder = OneHotEncoder(range(-2,81))
        label = winner_color*np.exp((1-number_moves_left)/9)*enocoder.fit_transform(next_move)

        if self.transform:
            Board = self.transform(Board)

        return fused_board, label


if __name__ == "__main__":
    DATASET_PATH = "./data/mini_dataset"
    D = Go9x9_Dataset(DATASET_PATH)
    