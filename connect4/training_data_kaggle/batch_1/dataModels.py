import typing
from pydantic import BaseModel, Field
import json
import os
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader

cwd = os.getcwd()

file_number = 1000

path = r"C:\Users\seany\Documents\ML\connect4\training_data_kaggle\batch_1"

class GameState(BaseModel):
    board: list[list[str]]
    current_player: str
    is_terminal: bool
    winner: str | None

class Overall_dataset(Dataset):
    def __init__(self):
        self.data_list = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        tensor, number = self.data_list[index]
        return tensor, number

with open(path) as f:
    data = json.load(f)

state_history = data['info']['stateHistory']

game_state = [ 
    GameState.model_validate_json(state)
    for state in data['info']['stateHistory']
]

def get_turn(turn_number, game_state: list):
    #keeping in mind that x player, or player one always goes first
    player_1_board = torch.zeros((6,7))
    player_2_board = torch.zeros((6,7))

    #figuring out which player just placed a piece
    #the player that just placed a piece is player placed
    if game_state[turn_number - 1].current_player == "x":
        player_placed_turn = torch.ones((6,7))
    elif game_state[turn_number - 1].current_player == "o":
        player_placed_turn = torch.zeros((6,7))

    #obtaining player piece matrices
    for row in range(0,6):
        for column in range(0,7):
            piece = game_state[turn_number].board[row][column]

            if piece == "x":
                player_1_board[row][column] = 1
            elif piece =="o":
                player_2_board[row][column] = 1
    
    #obtaining winning player 
    if game_state[len(game_state) - 1].winner == "x":
        winning_player = 1
    elif game_state[len(game_state) - 1].winner == "o":
        winning_player = 0
    elif game_state[len(game_state) - 1].winner == "draw":
        winning_player = 2
    else:
        winning_player = 3

    #player 1 board and player 2 boards are fliped across the x axis, or reflected horizontally here
    #as the original data seem to have the data from the bottom up
    #while pytorch seems to like to top down
    return torch.stack([player_1_board.flip(0), player_2_board.flip(0), player_placed_turn]), winning_player

output_dataset = Overall_dataset()
a = 0
for entry in pathlib.Path(cwd).iterdir():
    if entry.is_file() and entry.suffix not in {".py",".pt"}:
            a += 1
            print(a)
            print(entry)
            with open(entry) as f:
                data = json.load(f)
            state_history = data['info']['stateHistory']
            game_state = [ 
                GameState.model_validate_json(state)
                for state in data['info']['stateHistory']
            ]

            for i in range(1, len(game_state)):
                data = get_turn(i, game_state)
                if data[1] == 3:
                    break
                output_dataset.data_list.append(data)

        


torch.save(output_dataset, "compiled.pt")
