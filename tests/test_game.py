import numpy as np
import pytest

from gomoku_ai.game import (
    NUM_ACTIONS,
    PLAYER_BLACK,
    GomokuPosition,
)


def test_horizontal_five_terminates_game() -> None:
    position = GomokuPosition.initial()
    for action in (0, 15, 1, 16, 2, 17, 3, 18, 4):
        position = position.play(action)

    assert position.terminated
    assert position.winner == PLAYER_BLACK
    assert position.outcome_for(PLAYER_BLACK) == 1


def test_position_is_copied_when_playing() -> None:
    position = GomokuPosition.initial()
    child = position.play(112)

    assert np.count_nonzero(position.board) == 0
    assert child.board[112] == PLAYER_BLACK


@pytest.mark.parametrize("action", [-1, NUM_ACTIONS])
def test_rejects_out_of_range_action(action: int) -> None:
    with pytest.raises(ValueError):
        GomokuPosition.initial().play(action)
