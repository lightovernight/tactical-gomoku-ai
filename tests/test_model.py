import hashlib

import torch

from gomoku_ai.game import NUM_ACTIONS, GomokuPosition
from gomoku_ai.model import (
    DEFAULT_MODEL_PATH,
    canonical_boards_to_tensor,
    load_model,
)

EXPECTED_MODEL_SHA256 = (
    "7ce7a2f0ab8a9b49e2be0349041acce61c60cf732a0d1437996f7e40203e91da"
)


def test_model_checksum() -> None:
    digest = hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_MODEL_SHA256


def test_model_loads_and_has_expected_shapes() -> None:
    network, device, checkpoint = load_model(device=torch.device("cpu"))
    state = canonical_boards_to_tensor(
        GomokuPosition.initial().canonical_board(),
        device,
    )
    with torch.inference_mode():
        policy_logits, value = network(state)

    assert policy_logits.shape == (NUM_ACTIONS,)
    assert value.shape == ()
    assert checkpoint["games_completed"] == 25_000
    assert checkpoint["gradient_steps"] == 250_000
