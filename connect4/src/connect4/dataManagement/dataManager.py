import pathlib
from pathlib import Path
import torch
from torch.utils.data import Dataset, TensorDataset
from pydantic import BaseModel
import json

script_dir = pathlib.Path(__file__).resolve().parent

#a class that can handle the compiled data files along with torch dataloader    
class TurnOutcomeSet(Dataset):
    def __init__(self):
        self.data_list = []

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        tensor, number = self.data_list[index]
        return tensor, number
# a class used along with pydantic to interface with kaggle data
class GameState(BaseModel):
    board: list[list[str]]
    current_player: str
    is_terminal: bool
    winner: str | None


#helper methods in the function below these two
def get_result(dataset: tuple, game_outcome):
    player_that_placed = int(dataset[2][0][0])
    if game_outcome == player_that_placed:
        return 0       # win
    elif game_outcome == 2:
        return 2       # draw
    else:
        return 1       # loss
def get_data_path(batchn: int):
    script_dir = Path(__file__).resolve().parent
    return Path(str(Path(script_dir).parent.parent.parent) + r"/training_data_kaggle/batch_" + str(batchn) + r"/compiled.pt")

#the important method to get tensor dataset that uses the two methods above
def get_data(batch_n1, batch_n2):
    print("initiating data loading...")
    input_position_matrices = []
    game_outcomes = []
    for subBatchNumber in range(batch_n1, batch_n2 + 1):
        print("loading data for: batch" + str(subBatchNumber))
        data = torch.load(get_data_path(subBatchNumber), weights_only=False)

        for i in range(0, data.__len__()):
           input_position_matrices.append(data.__getitem__(i)[0])
           game_outcomes.append(get_result(data.__getitem__(i)[0], data.__getitem__(i)[1]))
        print("batch load complete.")
    print("dataloading complete")
    return TensorDataset(
        torch.stack(input_position_matrices), 
        torch.tensor(game_outcomes, dtype=torch.long)
        )

#the following are helper methods for compile_all_data method
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
def get_data_dir_path(batchn: int):
    script_dir = Path(__file__).resolve().parent
    return Path(str(Path(script_dir).parent.parent.parent) + r"/training_data_kaggle/batch_" + str(batchn))

def compile_all_data(batch_n1: int = 1, batch_n2: int = 11):
    script_dir = Path(__file__).resolve().parent
    print(f"[START] Compiling batches {batch_n1} to {batch_n2} (script dir: {script_dir})")
    for batch_index in range(batch_n1, batch_n2 + 1):
        output_dataset = TurnOutcomeSet()
        batch_dir = get_data_dir_path(batch_index)
        print(f"\n[BATCH {batch_index}] Starting batch, data dir: {batch_dir}")
        for entry in pathlib.Path(batch_dir).iterdir():
            if entry.is_file() and entry.suffix in {".py",".pt"}:
                print(f"[BATCH {batch_index}] Skipping {entry.name} (directory or .py/.pt file)")
                continue
            print(f"[BATCH {batch_index}] Processing file: {entry.name}")
            with open(entry) as f:
                data = json.load(f)
            state_history = data['info']['stateHistory']
            game_state = [ 
                GameState.model_validate_json(state)
                for state in data['info']['stateHistory']
            ]
            for turn in range(1, len(game_state)):
                data = get_turn(turn, game_state)
                if data[1] == 3:
                    break
                output_dataset.data_list.append(data)
        save_path = str(get_data_dir_path(batch_index)) + r"/compiled.pt"
        torch.save(output_dataset, Path(save_path))
        print(f"[BATCH {batch_index}] Saved successfully with {len(output_dataset.data_list)} turns to {save_path}")
