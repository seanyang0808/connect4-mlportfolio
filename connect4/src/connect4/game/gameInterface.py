import numpy as np

class ConnectFour:
    def __init__(self):
        self.board_width = 7
        self.board_height = 6
        self.board = np.zeros((self.board_height, self.board_width), dtype=int)
        self.gameover = False
        self.winner = None
        self._last_row = None
        self._last_col = None

    def reset(self):
        self.board.fill(0)
        self.gameover = False
        self.winner = None
        self._last_row = None
        self._last_col = None

    def check_move_legality(self, column_number: int):
        column_index_number = column_number - 1
        if 1 <= column_number <= self.board_width and self.board[0, column_index_number] == 0:
            return True
        else:
            legality = False
        return legality

    def drop_piece(self, column_number: int, player_number: int):
        column_index_number = column_number - 1
        if not self.check_move_legality(column_number):
            return ValueError
        else:
            for i in range(self.board_height - 1, -1, -1):
                if self.board[i, column_index_number] == 0:
                    self.board[i, column_index_number] = player_number
                    self._last_row = i
                    self._last_col = column_index_number
                    break
        if self.check_win(player_number):
            self.gameover = True
            self.winner = player_number
        return

    def check_win(self, player_number: int) -> bool:
        r, c = self._last_row, self._last_col
        if r is None or c is None:
            return False
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                for step in range(1, 4):
                    nr = r + sign * step * dr
                    nc = c + sign * step * dc
                    if 0 <= nr < self.board_height and 0 <= nc < self.board_width and self.board[nr, nc] == player_number:
                        count += 1
                    else:
                        break
            if count >= 4:
                return True
        return False

    def print_board(self):
        print(self.board)

