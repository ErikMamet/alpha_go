import torch
import numpy as np 
import os
import os.path as osp
import pickle
import torchvision.transforms as T
import time
from go_cnn import GoCNN

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


transf = T.ToTensor()

class Go9x9_Dataset(Dataset):
    def __init__(self, data_dir, size_of_input= 5, transform=transf, target_transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.nb_games = len(os.listdir(data_dir))
        #self.nb_games = 80000
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

        if last_player_color == -1:
            fused_board = np.zeros((1,9,9))
        if last_player_color == 1:
            fused_board = np.ones((1,9,9))
        
        t = time.time()

        for i in range(self.size_of_input):
            if (board_id - i > 0):
                board_path = osp.join(self.data_dir, "game_"+str(game_id), "state_"+str(board_id-i))
                with open(board_path, 'rb') as handle:
                    Old_Board, winner_color, last_player_color, old_next_move, number_moves_left = pickle.load(handle)
                    print(Old_Board)
                    print("next move", old_next_move)
                    Old_Board = np.reshape(Old_Board, (1,9,9))
                
                fused_board = np.vstack(( fused_board, Old_Board))
            else :
                fused_board = np.vstack(( fused_board, np.zeros((1,9,9)) ))

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

        #print("END IMPORT AFTER", time.time() - tt)
        return fused_board.type(torch.FloatTensor) , label


if __name__ == "__main__":
    model = GoCNN(6)
    model.load_state_dict(torch.load("log/80000_checkpoint_epoch_50"))
    DATASET_PATH = "./data/mini_dataset"
    D = Go9x9_Dataset(DATASET_PATH)
    item = D.__getitem__(5)
    print("---")
    print(item[1])
    print("---")
    policy, value = model(np.reshape(item[0],(1,6,9,9)))
    print(policy)
    print(value)
    print("max", torch.max(policy), "argmax", torch.argmax(policy))
    



    