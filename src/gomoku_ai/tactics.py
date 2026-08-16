from __future__ import annotations

from functools import lru_cache

import numpy as np

from .game import (
    BOARD_SIZE,
    EMPTY,
    NUM_ACTIONS,
    PLAYER_BLACK,
    PLAYER_WHITE,
    WIN_LENGTH,
    GomokuPosition,
)


@lru_cache(maxsize=1)
def _winning_windows() -> np.ndarray:
    windows: list[list[int]] = []
    for row_step, column_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for row in range(BOARD_SIZE):
            for column in range(BOARD_SIZE):
                end_row = row + (WIN_LENGTH - 1) * row_step
                end_column = column + (WIN_LENGTH - 1) * column_step
                if not (0 <= end_row < BOARD_SIZE and 0 <= end_column < BOARD_SIZE):
                    continue
                windows.append(
                    [
                        (row + offset * row_step) * BOARD_SIZE
                        + column
                        + offset * column_step
                        for offset in range(WIN_LENGTH)
                    ]
                )
    return np.asarray(windows, dtype=np.int64)


def analyze_one_ply(position: GomokuPosition) -> tuple[np.ndarray, np.ndarray]:
    if position.terminated:
        raise ValueError("Cannot analyze a terminal position")
    windows = _winning_windows()
    values = position.board[windows]
    own_wins = _winning_actions(
        windows,
        values,
        position.current_player,
    )
    opponent_wins = _winning_actions(
        windows,
        values,
        -position.current_player,
    )
    target_q = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    legal_actions = position.legal_actions
    if len(opponent_wins) == 1:
        forced_losses = legal_actions[legal_actions != opponent_wins[0]]
        target_q[forced_losses] = -1.0
        target_mask[forced_losses] = True
    elif len(opponent_wins) >= 2:
        target_q[legal_actions] = -1.0
        target_mask[legal_actions] = True
    target_q[own_wins] = 1.0
    target_mask[own_wins] = True
    return target_q, target_mask


def analyze_one_ply_batch(
    positions: list[GomokuPosition],
) -> tuple[np.ndarray, np.ndarray]:
    if not positions or any(position.terminated for position in positions):
        raise ValueError("Expected non-terminal tactical positions")
    boards = np.stack([position.board for position in positions])
    players = np.asarray(
        [position.current_player for position in positions],
        dtype=np.int8,
    )
    canonical = boards * players[:, None]
    windows = _winning_windows()
    values = canonical[:, windows]
    empty_counts = np.count_nonzero(values == EMPTY, axis=2)
    own_completes = (
        np.count_nonzero(values == PLAYER_BLACK, axis=2) == WIN_LENGTH - 1
    ) & (empty_counts == 1)
    opponent_completes = (
        np.count_nonzero(values == PLAYER_WHITE, axis=2) == WIN_LENGTH - 1
    ) & (empty_counts == 1)

    batch_size = len(positions)

    def completing_action_mask(completes: np.ndarray) -> np.ndarray:
        action_mask = np.zeros((batch_size, NUM_ACTIONS), dtype=bool)
        batch_indices, window_indices = np.nonzero(completes)
        if len(batch_indices):
            empty_offsets = np.argmax(
                values[batch_indices, window_indices] == EMPTY,
                axis=1,
            )
            completing_actions = windows[window_indices, empty_offsets]
            action_mask[batch_indices, completing_actions] = True
        return action_mask

    own_wins = completing_action_mask(own_completes)
    opponent_wins = completing_action_mask(opponent_completes)
    legal = boards == EMPTY
    opponent_win_counts = opponent_wins.sum(axis=1)
    one_threat = opponent_win_counts == 1
    multiple_threats = opponent_win_counts >= 2
    target_q = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
    target_mask = np.zeros((batch_size, NUM_ACTIONS), dtype=bool)
    if bool(one_threat.any()):
        forced_losses = legal[one_threat] & ~opponent_wins[one_threat]
        target_q[one_threat] = np.where(forced_losses, -1.0, 0.0)
        target_mask[one_threat] = forced_losses
    if bool(multiple_threats.any()):
        target_q[multiple_threats] = np.where(
            legal[multiple_threats],
            -1.0,
            0.0,
        )
        target_mask[multiple_threats] = legal[multiple_threats]
    target_q[own_wins] = 1.0
    target_mask[own_wins] = True
    return target_q, target_mask


def _winning_actions(
    windows: np.ndarray,
    window_values: np.ndarray,
    player: int,
) -> np.ndarray:
    completes = (
        np.count_nonzero(window_values == player, axis=1) == WIN_LENGTH - 1
    ) & (np.count_nonzero(window_values == EMPTY, axis=1) == 1)
    if not bool(completes.any()):
        return np.empty(0, dtype=np.int64)
    completing_windows = windows[completes]
    completing_values = window_values[completes]
    return np.unique(completing_windows[completing_values == EMPTY])
