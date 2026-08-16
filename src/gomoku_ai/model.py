from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .game import BOARD_SIZE, NUM_ACTIONS

CHANNELS = 64
RESIDUAL_BLOCKS = 4
VALUE_HIDDEN_SIZE = 128
MODEL_FORMAT_VERSION = 1
MODEL_TYPE = "gomoku_policy_value_release"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "gomoku_25000.pt"


def as_state_batch(states: torch.Tensor, board_size: int) -> torch.Tensor:
    batched = states.unsqueeze(0) if states.dim() == 3 else states
    if (
        batched.dim() != 4
        or batched.size(1) != 3
        or batched.size(2) != board_size
        or batched.size(3) != board_size
    ):
        raise ValueError("states must have shape [batch, 3, 15, 15]")
    return batched


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(states)))
        return self.relu(states + residual)


class ResidualPolicyValueNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.board_size = BOARD_SIZE
        self.num_actions = NUM_ACTIONS
        self.stem = nn.Sequential(
            nn.Conv2d(3, CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(RESIDUAL_BLOCKS)))
        self.policy_head = nn.Conv2d(CHANNELS, 1, kernel_size=1)
        self.value_features = nn.Sequential(
            nn.Conv2d(CHANNELS, 1, kernel_size=1),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(NUM_ACTIONS, VALUE_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(VALUE_HIDDEN_SIZE, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_state = states.dim() == 3
        batched = as_state_batch(states, self.board_size)
        features = self.blocks(self.stem(batched))
        policy_logits = self.policy_head(features).flatten(1)
        values = torch.tanh(
            self.value_head(self.value_features(features).flatten(1))
        ).squeeze(1)
        if single_state:
            return policy_logits[0], values[0]
        return policy_logits, values


def canonical_boards_to_tensor(
    canonical_boards: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    boards = np.asarray(canonical_boards)
    single_board = boards.ndim == 1
    if single_board:
        boards = boards[None, :]
    if boards.ndim != 2 or boards.shape[1] != NUM_ACTIONS:
        raise ValueError(f"canonical boards must contain {NUM_ACTIONS} cells")

    matrices = boards.reshape(-1, BOARD_SIZE, BOARD_SIZE)
    channels = np.empty(
        (len(matrices), 3, BOARD_SIZE, BOARD_SIZE),
        dtype=np.float32,
    )
    channels[:, 0] = matrices == 1
    channels[:, 1] = matrices == -1
    channels[:, 2] = matrices == 0
    if single_board:
        channels = channels[0]
    return torch.as_tensor(channels, dtype=torch.float32, device=device)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    device: torch.device | None = None,
) -> tuple[nn.Module, torch.device, dict]:
    selected_device = resolve_device() if device is None else device
    checkpoint = torch.load(
        Path(model_path),
        map_location=selected_device,
        weights_only=True,
    )
    if checkpoint.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported release model format")
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError("The file is not a Gomoku release model")

    network = ResidualPolicyValueNetwork().to(selected_device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        inference_network = torch.jit.script(network)
    inference_network.eval()
    return inference_network, selected_device, checkpoint
