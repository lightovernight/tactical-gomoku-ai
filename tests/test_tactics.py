import numpy as np

from gomoku_ai.game import (
    NUM_ACTIONS,
    PLAYER_BLACK,
    PLAYER_WHITE,
    GomokuPosition,
)
from gomoku_ai.tactics import analyze_one_ply, analyze_one_ply_batch


def make_position(stones: dict[int, int], current_player: int) -> GomokuPosition:
    board = np.zeros(NUM_ACTIONS, dtype=np.int8)
    for action, player in stones.items():
        board[action] = player
    return GomokuPosition(
        board=board,
        current_player=current_player,
        move_count=len(stones),
    )


def test_immediate_win_is_proved() -> None:
    position = make_position(
        {0: PLAYER_BLACK, 1: PLAYER_BLACK, 2: PLAYER_BLACK, 3: PLAYER_BLACK},
        PLAYER_BLACK,
    )
    values, mask = analyze_one_ply(position)

    assert mask[4]
    assert values[4] == 1.0


def test_only_forced_block_is_not_marked_as_loss() -> None:
    position = make_position(
        {0: PLAYER_WHITE, 1: PLAYER_WHITE, 2: PLAYER_WHITE, 3: PLAYER_WHITE},
        PLAYER_BLACK,
    )
    values, mask = analyze_one_ply(position)

    legal = position.legal_mask.astype(bool)
    expected_losses = legal.copy()
    expected_losses[4] = False
    assert np.array_equal(mask, expected_losses)
    assert np.all(values[expected_losses] == -1.0)


def test_batch_analysis_matches_single_analysis() -> None:
    positions = [
        make_position(
            {0: PLAYER_BLACK, 1: PLAYER_BLACK, 2: PLAYER_BLACK, 3: PLAYER_BLACK},
            PLAYER_BLACK,
        ),
        make_position(
            {15: PLAYER_WHITE, 30: PLAYER_WHITE, 45: PLAYER_WHITE, 60: PLAYER_WHITE},
            PLAYER_BLACK,
        ),
    ]
    batch_values, batch_masks = analyze_one_ply_batch(positions)

    for index, position in enumerate(positions):
        values, mask = analyze_one_ply(position)
        assert np.array_equal(batch_values[index], values)
        assert np.array_equal(batch_masks[index], mask)
