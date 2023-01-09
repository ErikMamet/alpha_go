from sgfmill import sgf
import os
import Goban
import pickle
import copy

path = "/home/lucasbardis/Downloads/Go_Dataset/gokif3"

def get_moves(file_name:str):
    '''Get moves from a SGF file'''
    with open("{}/{}".format(path,file_name), "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
    moves = []
    for node in game.get_main_sequence():
        if (node.get_move() != (None, None)):
            if (check_move(node.get_move()[1])):
                if node.get_move()[1] == None: moves.append(-1)
                else: moves.append(Goban.Board.flatten(node.get_move()[1]))
            else:
                return False, []
    return True, moves

def check_move(move: tuple):
    '''Check if move is valid'''
    if(move == None): return True
    if (move[0] > 8 or move[1] > 8): return False
    return True

def create_game_boards(moves: list):
    '''Create a board if possible'''
    board = Goban.Board()
    remaining_moves = len(moves)
    color_playing = 1
    all_states = [[board, None, color_playing, moves[0], remaining_moves]]
    for move in moves:
        color_playing = color_playing * -1
        valid = board.push(move) 
        if not valid:
            return False, None
        remaining_moves += -1
        if (remaining_moves == 0): 
            next_move = -2
            winner = get_winner(board.result())
        else: next_move = moves[len(moves)-remaining_moves]
        all_states.append([copy.deepcopy(board._board), None, color_playing, next_move, remaining_moves])
    for state in all_states:
        state[1] = winner
    return True, all_states

def get_winner(result: str):
    if (result == "1-0"): winner = 1 #White wins
    elif (result == "0-1"): winner = -1 #Black wins
    else: winner = 0 #Deuce 
    return winner

def create_all_boards():
    '''Create all valid boards'''
    folder = os.listdir(path)
    invalid_moves = 0
    invalid_games = 0
    valid_games = 0
    i = 0
    for file in folder:
        if (i % 100 == 0):
            print(i)
            print("Valid boards : ", valid_games)
            print("Invalid boards : ", invalid_games)
            print("Invalid moves : ", invalid_moves)
        are_moves_valid, moves = get_moves(file)
        if (are_moves_valid):
            is_game_valid = True
            is_game_valid, all_states = create_game_boards(moves)
            if (is_game_valid):
                valid_games += 1
                for j in range(len(all_states)):
                    save_board(all_states[j], valid_games, j)
            else:
                invalid_games += 1
        else: 
            invalid_moves += 1
        i = i + 1
    print("Valid boards : ", valid_games)
    print("Invalid boards : ", invalid_games)
    print("Invalid moves : ", invalid_moves)
    return valid_games, invalid_games, invalid_moves

def save_board(board: Goban.Board, n_game: str, n_state: str):
    '''Save one board with pickle'''
    filename = "data/dataset/game_{}/state_{}".format(n_game, n_state)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with(open(filename, "wb" )) as fp:
        pickle.dump(board, fp)

create_all_boards()