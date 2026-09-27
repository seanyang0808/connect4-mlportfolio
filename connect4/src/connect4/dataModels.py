import typing
from pydantic import BaseModel, Field
import json
import os
import pathlib
import torch

cwd = os.getcwd()

file_number = 1000

path = r"C:\dev\ML\connect4\training_data_kaggle\batch_1\training_data_kagglebatch_1085103561.json"

class GameState(BaseModel):
    board: list[list[str]]
    current_player: str
    is_terminal: bool
    winner: str | None

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
        player_placed_board = torch.ones((6,7))
    elif game_state[turn_number - 1].current_player == "o":
        player_placed_board = torch.zeros((6,7))

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

    #player 1 board and player 2 boards are fliped across the x axis, or reflected horizontally here
    #as the original data seem to have the data from the bottom up
    #while pytorch seems to like to top down
    return torch.stack([player_1_board.flip(0), player_2_board.flip(0), player_placed_board]), winning_player
    
