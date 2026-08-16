from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

PLAYER_BLACK = 1
PLAYER_WHITE = -1
EMPTY = 0
BOARD_SIZE = 15
WIN_LENGTH = 5
NUM_ACTIONS = BOARD_SIZE * BOARD_SIZE


@dataclass(slots=True)
class GomokuPosition:
    """Immutable-by-convention position for fixed 15x15 freestyle Gomoku."""

    board: np.ndarray
    current_player: int
    winner: int | None = None
    last_action: int | None = None
    move_count: int = 0
    terminated: bool = False

    @classmethod
    def initial(cls) -> GomokuPosition:
        return cls(
            board=np.zeros(NUM_ACTIONS, dtype=np.int8),
            current_player=PLAYER_BLACK,
        )

    @property
    def board_size(self) -> int:
        return BOARD_SIZE

    @property
    def win_length(self) -> int:
        return WIN_LENGTH

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def legal_mask(self) -> np.ndarray:
        if self.terminated:
            return np.zeros(NUM_ACTIONS, dtype=np.int8)
        return (self.board == EMPTY).astype(np.int8)

    @property
    def legal_actions(self) -> np.ndarray:
        if self.terminated:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.board == EMPTY)

    def canonical_board(self) -> np.ndarray:
        return (self.board * self.current_player).astype(np.int8, copy=False)

    def play(self, action: int) -> GomokuPosition:
        if self.terminated:
            raise RuntimeError("Cannot play from a terminal position")
        action = _validate_action(action)
        if self.board[action] != EMPTY:
            raise ValueError(f"Position {action} is already occupied")

        acting_player = self.current_player
        next_board = self.board.copy()
        next_board[action] = acting_player
        next_move_count = self.move_count + 1
        if _has_winning_line(next_board, action, acting_player):
            winner: int | None = acting_player
            terminated = True
        elif next_move_count == NUM_ACTIONS:
            winner = EMPTY
            terminated = True
        else:
            winner = None
            terminated = False

        return GomokuPosition(
            board=next_board,
            current_player=-acting_player,
            winner=winner,
            last_action=action,
            move_count=next_move_count,
            terminated=terminated,
        )

    def outcome_for(self, player: int) -> int:
        if player not in {PLAYER_BLACK, PLAYER_WHITE}:
            raise ValueError(f"Invalid player: {player}")
        if not self.terminated or self.winner is None:
            raise RuntimeError("The position is not terminal")
        return int(self.winner * player)

    def copy(self) -> GomokuPosition:
        return GomokuPosition(
            board=self.board.copy(),
            current_player=self.current_player,
            winner=self.winner,
            last_action=self.last_action,
            move_count=self.move_count,
            terminated=self.terminated,
        )


def _validate_action(action: int) -> int:
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError(f"Action must be an integer, got {type(action).__name__}")
    action = int(action)
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"Action must be in [0, {NUM_ACTIONS - 1}], got {action}")
    return action


def _has_winning_line(
    board: np.ndarray,
    last_action: int,
    player: int,
) -> bool:
    row, column = divmod(last_action, BOARD_SIZE)
    matrix = board.reshape(BOARD_SIZE, BOARD_SIZE)
    for row_step, column_step in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        count += _count_direction(
            matrix,
            row,
            column,
            row_step,
            column_step,
            player,
        )
        count += _count_direction(
            matrix,
            row,
            column,
            -row_step,
            -column_step,
            player,
        )
        if count >= WIN_LENGTH:
            return True
    return False


def _count_direction(
    board: np.ndarray,
    row: int,
    column: int,
    row_step: int,
    column_step: int,
    player: int,
) -> int:
    count = 0
    row += row_step
    column += column_step
    while (
        0 <= row < BOARD_SIZE
        and 0 <= column < BOARD_SIZE
        and board[row, column] == player
    ):
        count += 1
        row += row_step
        column += column_step
    return count
