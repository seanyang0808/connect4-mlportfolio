import argparse
import sys

import numpy as np
import torch

from connect4.game.gameInterface import ConnectFour
from connect4.modelHandling.model import CNN

from pathlib import Path

script_dir = Path(__file__).resolve().parent

MODEL_PATH = str(script_dir.parent.parent) + "/models/testModel_280926.pt"

WIN, LOSS, DRAW = 0, 1, 2


class ModelPlayer:
    def __init__(self, model_path: str = MODEL_PATH, score_mode: str = "win-loss"):
        self.model = CNN()
        self.model.load_model(model_path)
        self.model.eval()
        self.score_mode = score_mode

    def _to_tensor(self, board: np.ndarray, player: int) -> torch.Tensor:
        player_one = torch.tensor((board == 1).astype(np.float32))
        player_two = torch.tensor((board == 2).astype(np.float32))
        placed = torch.full((6, 7), 1.0) if player == 1 else torch.zeros((6, 7))
        return torch.stack([player_one, player_two, placed]).unsqueeze(0)

    def predict(self, board: np.ndarray, player: int) -> list[float]:
        with torch.no_grad():
            logits = self.model(self._to_tensor(board, player))
            probs = torch.softmax(logits, dim=1)[0]
        return probs.tolist()

    def _score(self, probs: list[float]) -> float:
        if self.score_mode in ("win", "expected"):
            return probs[WIN]
        return probs[WIN] - probs[LOSS]

    def move(self, game: ConnectFour, player: int, verbose: bool = False):
        best_col, best_score = None, -float("inf")
        results = {}
        for col in range(1, 8):
            if not game.check_move_legality(col):
                continue
            sim = ConnectFour()
            sim.board = game.board.copy()
            sim.drop_piece(col, player)
            if sim.gameover:
                return col, [1.0, 0.0, 0.0]
            probs = self.predict(sim.board, player)
            score = self._score(probs)
            results[col] = (probs, score)
            if score > best_score or (score == best_score and abs(col - 4) < abs(best_col - 4)):
                best_col, best_score = col, score
        if verbose and best_col is not None:
            for col in sorted(results):
                probs, score = results[col]
                print(
                    f"  col {col}: win={probs[WIN]:.3f} loss={probs[LOSS]:.3f} "
                    f"draw={probs[DRAW]:.3f} score={score:+.3f}"
                )
        probs = results[best_col][0] if best_col in results else [1.0, 0.0, 0.0]
        return best_col, probs


def render(game: ConnectFour) -> None:
    print("\n  " + " ".join(str(c) for c in range(1, 8)))
    for row in game.board:
        print("  " + " ".join("X" if v == 1 else "O" if v == 2 else "." for v in row))


def choose_color() -> int:
    while True:
        choice = input("Play as X (first) or O (second)? [X/o]: ").strip().lower()
        if choice in ("x", ""):
            return 1
        if choice == "o":
            return 2


def human_turn(game: ConnectFour, player: int) -> None:
    while True:
        raw = input(f"Your move (1-7) [{('X' if player == 1 else 'O')}]: ").strip()
        if not raw.isdigit() or not game.check_move_legality(int(raw)):
            print("Invalid column. Pick 1-7 with an open slot.")
            continue
        game.drop_piece(int(raw), player)
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Connect Four against a trained CNN model.")
    parser.add_argument(
        "--score",
        choices=["win-loss", "win", "expected"],
        default="win-loss",
        help="move scoring mode (default: win-loss)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the model's win/loss/draw probabilities during the AI turn",
    )
    args = parser.parse_args()

    ai = ModelPlayer(score_mode=args.score)

    while True:
        game = ConnectFour()
        human_player = choose_color()
        ai_player = 3 - human_player
        turn = 1
        print("\n" + ("You are X and go first." if human_player == 1 else "You are O; the AI goes first."))

        while not game.gameover:
            render(game)
            if turn == human_player:
                human_turn(game, human_player)
            else:
                col, _ = ai.move(game, ai_player, verbose=args.verbose)
                print(f"\nAI plays column {col}.")
                game.drop_piece(col, ai_player)

            if game.gameover:
                render(game)
                if game.winner is not None:
                    print(("You win!" if game.winner == human_player else "AI wins."))
                else:
                    print("It's a draw.")
                break

            if game.board.all():
                render(game)
                print("It's a draw.")
                break

            turn = 3 - turn

        again = input("\nPlay again? [Y/n]: ").strip().lower()
        if again == "n":
            print("Goodbye.")
            sys.exit(0)


if __name__ == "__main__":
    main()
