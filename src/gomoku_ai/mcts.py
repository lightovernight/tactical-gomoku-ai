from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import torch
from torch import nn

from .game import NUM_ACTIONS, GomokuPosition
from .model import canonical_boards_to_tensor
from .tactics import analyze_one_ply, analyze_one_ply_batch

DEFAULT_SIMULATIONS = 6400
DEFAULT_BATCH_SIZE = 64
EXPLORATION_CONSTANT = 1.5

_PROOF_UNKNOWN = np.int8(0)
_PROOF_WIN = np.int8(1)
_PROOF_LOSS = np.int8(-1)


@dataclass(slots=True)
class SearchResult:
    q_values: np.ndarray
    visit_counts: np.ndarray
    legal_mask: np.ndarray
    proven_q_values: np.ndarray
    proven_mask: np.ndarray
    policy_prior: np.ndarray
    simulations_added: int

    @property
    def searched_mask(self) -> np.ndarray:
        return self.visit_counts > 0


@dataclass(slots=True)
class _Node:
    position: GomokuPosition
    expanded: bool = False
    tactical_proof_scanned: bool = False
    visit_counts: np.ndarray = field(init=False)
    total_values: np.ndarray = field(init=False)
    virtual_visit_counts: np.ndarray = field(init=False)
    prior_probabilities: np.ndarray = field(init=False)
    network_prior_probabilities: np.ndarray = field(init=False)
    proof_status: np.ndarray = field(init=False)
    legal_mask: np.ndarray = field(init=False)
    legal_actions: np.ndarray = field(init=False)
    candidate_mask: np.ndarray = field(init=False)
    candidate_prior_scale: float = 1.0
    children: dict[int, _Node] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.visit_counts = np.zeros(NUM_ACTIONS, dtype=np.int32)
        self.total_values = np.zeros(NUM_ACTIONS, dtype=np.float64)
        self.virtual_visit_counts = np.zeros(NUM_ACTIONS, dtype=np.int32)
        self.prior_probabilities = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.network_prior_probabilities = np.zeros(
            NUM_ACTIONS,
            dtype=np.float32,
        )
        self.proof_status = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self.legal_mask = self.position.legal_mask.astype(bool)
        self.legal_actions = np.flatnonzero(self.legal_mask)
        self.candidate_mask = self.legal_mask.copy()

    @property
    def proven_mask(self) -> np.ndarray:
        return self.proof_status != _PROOF_UNKNOWN

    @property
    def proven_state_value(self) -> float | None:
        legal_status = self.proof_status[self.legal_actions]
        if bool(np.any(legal_status == _PROOF_WIN)):
            return 1.0
        if len(legal_status) and bool(np.all(legal_status == _PROOF_LOSS)):
            return -1.0
        return None

    def set_proven_action_values(
        self,
        target_q: np.ndarray,
        target_mask: np.ndarray,
    ) -> None:
        mask = np.asarray(target_mask, dtype=bool) & self.legal_mask
        values = np.asarray(target_q, dtype=np.float32)
        if values.shape != self.proof_status.shape or mask.shape != values.shape:
            raise ValueError("Tactical proof arrays must match the action shape")
        invalid = mask & (values != 1.0) & (values != -1.0)
        if bool(invalid.any()):
            raise ValueError("Tactical proofs must be exactly +1 or -1")

        new_status = np.zeros_like(self.proof_status)
        new_status[mask & (values == 1.0)] = _PROOF_WIN
        new_status[mask & (values == -1.0)] = _PROOF_LOSS
        contradiction = mask & self.proven_mask & (self.proof_status != new_status)
        if bool(contradiction.any()):
            raise AssertionError("Contradictory tactical proofs")
        self.proof_status[mask] = new_status[mask]
        self._refresh_candidates()

    def set_proven_action(self, action: int, value: float) -> None:
        if not self.legal_mask[action]:
            raise ValueError("Cannot prove an illegal action")
        if value not in {-1.0, 1.0}:
            raise ValueError("A proof must be exactly +1 or -1")
        status = _PROOF_WIN if value == 1.0 else _PROOF_LOSS
        existing = self.proof_status[action]
        if existing not in {_PROOF_UNKNOWN, status}:
            raise AssertionError("Contradictory propagated proofs")
        self.proof_status[action] = status
        self._refresh_candidates()

    def _refresh_candidates(self) -> None:
        wins = self.legal_mask & (self.proof_status == _PROOF_WIN)
        non_losses = self.legal_mask & (self.proof_status != _PROOF_LOSS)
        if bool(wins.any()):
            self.candidate_mask = wins
        elif bool(non_losses.any()):
            self.candidate_mask = non_losses
        else:
            self.candidate_mask = self.legal_mask.copy()
        self._refresh_candidate_prior_scale()

    def _refresh_candidate_prior_scale(self) -> None:
        candidate_total = float(self.prior_probabilities[self.candidate_mask].sum())
        self.candidate_prior_scale = (
            1.0 / candidate_total if candidate_total > 0 else 1.0
        )

    def expand(self, probabilities: np.ndarray) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.shape != self.prior_probabilities.shape:
            raise ValueError("Policy prior has the wrong shape")
        if not np.isfinite(probabilities[self.legal_mask]).all():
            raise ValueError("Policy prior contains a non-finite value")
        if np.any(probabilities[~self.legal_mask] != 0):
            raise ValueError("Policy prior assigns mass to an illegal action")
        legal_total = float(probabilities[self.legal_mask].sum())
        if not np.isclose(legal_total, 1.0, atol=1e-5):
            raise ValueError(f"Policy prior sums to {legal_total}, not one")
        self.network_prior_probabilities[:] = probabilities
        self.prior_probabilities[:] = probabilities
        self.expanded = True
        self._refresh_candidate_prior_scale()

    def q_values(self) -> np.ndarray:
        values = np.zeros(NUM_ACTIONS, dtype=np.float32)
        np.divide(
            self.total_values,
            self.visit_counts,
            out=values,
            where=self.visit_counts > 0,
        )
        proved = self.proven_mask
        values[proved] = self.proof_status[proved].astype(np.float32)
        return values


@dataclass(slots=True)
class _PendingSimulation:
    path: list[tuple[_Node, int]]
    leaf: _Node | None
    terminal_value: float | None


class TacticalPUCT:
    """Fixed policy-value PUCT used by the release agent."""

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        *,
        simulations: int = DEFAULT_SIMULATIONS,
        inference_batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 0,
    ) -> None:
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size must be positive")
        self.network = network
        self.device = device
        self.simulations = int(simulations)
        self.inference_batch_size = int(inference_batch_size)
        self.rng = np.random.default_rng(seed)
        self._root: _Node | None = None
        self._score_buffer: np.ndarray | None = None
        self._tie_buffer: np.ndarray | None = None

    def run(self, position: GomokuPosition) -> SearchResult:
        if position.terminated:
            raise ValueError("Cannot search a terminal position")
        root = self._ensure_root(position)
        if not root.tactical_proof_scanned:
            self._apply_tactical_proofs(root)
        if not root.expanded:
            self._evaluate_and_expand(root, expand_proved_node=True)

        existing_visits = int(root.visit_counts.sum())
        simulations_added = max(0, self.simulations - existing_visits)
        if self.inference_batch_size == 1:
            for _ in range(simulations_added):
                self._simulate(root)
        else:
            remaining = simulations_added
            while remaining:
                current = min(self.inference_batch_size, remaining)
                self._simulate_batch(root, current)
                remaining -= current

        repair_simulations = 0
        while not bool(np.any(root.visit_counts[root.candidate_mask] > 0)):
            self._simulate(root)
            simulations_added += 1
            repair_simulations += 1
            if repair_simulations > NUM_ACTIONS:
                raise AssertionError("Proof-aware root repair did not converge")

        return SearchResult(
            q_values=root.q_values(),
            visit_counts=root.visit_counts.copy(),
            legal_mask=root.legal_mask.copy(),
            proven_q_values=root.proof_status.astype(np.float32, copy=True),
            proven_mask=root.proven_mask.copy(),
            policy_prior=root.network_prior_probabilities.copy(),
            simulations_added=simulations_added,
        )

    def advance_root(self, action: int) -> bool:
        if self._root is None:
            return False
        child = self._root.children.get(action)
        reused = child is not None
        if child is None:
            child = _Node(self._root.position.play(action))
        self._root = child
        return reused

    def reset(self) -> None:
        self._root = None

    @property
    def retained_root_visits(self) -> int:
        if self._root is None:
            return 0
        return int(self._root.visit_counts.sum())

    def _ensure_root(self, position: GomokuPosition) -> _Node:
        if self._root is None or not self._same_position(
            self._root.position,
            position,
        ):
            self._root = _Node(position.copy())
        return self._root

    @staticmethod
    def _same_position(left: GomokuPosition, right: GomokuPosition) -> bool:
        return (
            left.current_player == right.current_player
            and left.winner == right.winner
            and left.terminated == right.terminated
            and np.array_equal(left.board, right.board)
        )

    def _simulate(self, root: _Node) -> None:
        pending = self._traverse(root, reserve_virtual=False)
        leaf_value = (
            pending.terminal_value
            if pending.leaf is None
            else self._evaluate_and_expand(pending.leaf)
        )
        self._backup(pending.path, float(leaf_value))
        if pending.leaf is None or pending.leaf.proven_state_value is not None:
            self._propagate_proof(pending.path, float(leaf_value))

    def _simulate_batch(self, root: _Node, batch_size: int) -> None:
        pending: list[_PendingSimulation] = []
        unique_leaves: list[_Node] = []
        seen_leaves: set[int] = set()
        for _ in range(batch_size):
            simulation = self._traverse(root, reserve_virtual=True)
            pending.append(simulation)
            if simulation.leaf is not None and id(simulation.leaf) not in seen_leaves:
                seen_leaves.add(id(simulation.leaf))
                unique_leaves.append(simulation.leaf)

        leaf_values = self._evaluate_and_expand_batch(unique_leaves)
        value_by_leaf = {
            id(leaf): float(value)
            for leaf, value in zip(unique_leaves, leaf_values, strict=True)
        }
        for simulation in pending:
            value = (
                float(simulation.terminal_value)
                if simulation.leaf is None
                else value_by_leaf[id(simulation.leaf)]
            )
            self._commit_virtual_backup(simulation.path, value)
            if (
                simulation.leaf is None
                or simulation.leaf.proven_state_value is not None
            ):
                self._propagate_proof(simulation.path, value)

    def _traverse(
        self,
        root: _Node,
        *,
        reserve_virtual: bool,
    ) -> _PendingSimulation:
        node = root
        path: list[tuple[_Node, int]] = []
        while True:
            if node.position.terminated:
                terminal_value = float(
                    node.position.outcome_for(node.position.current_player)
                )
                leaf = None
                break
            proved_value = node.proven_state_value
            if path and proved_value is not None:
                terminal_value = proved_value
                leaf = None
                break
            if not node.expanded:
                terminal_value = None
                leaf = node
                break

            action = self._select_tree_action(node)
            path.append((node, action))
            child = node.children.get(action)
            if child is None:
                child = _Node(node.position.play(action))
                node.children[action] = child
            node = child

        if reserve_virtual:
            for parent, action in path:
                parent.virtual_visit_counts[action] += 1
        return _PendingSimulation(path, leaf, terminal_value)

    def _select_tree_action(self, node: _Node) -> int:
        if self._score_buffer is None:
            self._score_buffer = np.empty(NUM_ACTIONS, dtype=np.float64)
            self._tie_buffer = np.empty(NUM_ACTIONS, dtype=bool)
        assert self._tie_buffer is not None

        effective_counts = node.visit_counts + node.virtual_visit_counts
        parent_visits = max(1, int(effective_counts.sum()))
        q_values = node.q_values()
        np.divide(
            EXPLORATION_CONSTANT
            * node.prior_probabilities
            * node.candidate_prior_scale
            * sqrt(parent_visits),
            1.0 + effective_counts,
            out=self._score_buffer,
        )
        np.add(q_values, self._score_buffer, out=self._score_buffer)
        self._score_buffer[~node.candidate_mask] = -np.inf
        best_action = int(self._score_buffer.argmax())
        np.equal(
            self._score_buffer,
            self._score_buffer[best_action],
            out=self._tie_buffer,
        )
        best_actions = np.flatnonzero(self._tie_buffer)
        if len(best_actions) == 1:
            return best_action
        return int(best_actions[int(self.rng.integers(len(best_actions)))])

    def _evaluate_and_expand(
        self,
        node: _Node,
        *,
        expand_proved_node: bool = False,
    ) -> float:
        proved_value = self._apply_tactical_proofs(node)
        if proved_value is not None and not expand_proved_node:
            return proved_value

        state = canonical_boards_to_tensor(
            node.position.canonical_board(),
            self.device,
        )
        with torch.inference_mode():
            policy_logits, value = self.network(state)
            legal_mask = torch.as_tensor(
                node.legal_mask,
                dtype=torch.bool,
                device=self.device,
            )
            probabilities = torch.softmax(
                policy_logits.masked_fill(~legal_mask, -torch.inf),
                dim=0,
            )
        node.expand(probabilities.detach().cpu().numpy().astype(np.float32, copy=False))
        return proved_value if proved_value is not None else float(value.item())

    def _evaluate_and_expand_batch(self, nodes: list[_Node]) -> np.ndarray:
        if not nodes:
            return np.empty(0, dtype=np.float32)

        self._apply_tactical_proofs_batch(nodes)
        leaf_values = np.empty(len(nodes), dtype=np.float32)
        network_indices = [
            index for index, node in enumerate(nodes) if node.proven_state_value is None
        ]
        for index, node in enumerate(nodes):
            proved_value = node.proven_state_value
            if proved_value is not None:
                leaf_values[index] = proved_value
        if not network_indices:
            return leaf_values

        network_nodes = [nodes[index] for index in network_indices]
        boards = np.stack([node.position.canonical_board() for node in network_nodes])
        states = canonical_boards_to_tensor(boards, self.device)
        with torch.inference_mode():
            policy_logits, values = self.network(states)
            legal_masks = torch.as_tensor(
                np.stack([node.legal_mask for node in network_nodes]),
                dtype=torch.bool,
                device=self.device,
            )
            probabilities = torch.softmax(
                policy_logits.masked_fill(~legal_masks, -torch.inf),
                dim=1,
            )
        batch_probabilities = (
            probabilities.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )
        for node, prior in zip(
            network_nodes,
            batch_probabilities,
            strict=True,
        ):
            node.expand(prior)
        batch_values = (
            values.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )
        leaf_values[network_indices] = batch_values
        return leaf_values

    @staticmethod
    def _apply_tactical_proofs(node: _Node) -> float | None:
        if node.tactical_proof_scanned:
            return node.proven_state_value
        target_q, target_mask = analyze_one_ply(node.position)
        node.set_proven_action_values(target_q, target_mask)
        node.tactical_proof_scanned = True
        return node.proven_state_value

    @staticmethod
    def _apply_tactical_proofs_batch(nodes: list[_Node]) -> None:
        unscanned = [node for node in nodes if not node.tactical_proof_scanned]
        if not unscanned:
            return
        target_q, target_mask = analyze_one_ply_batch(
            [node.position for node in unscanned]
        )
        for index, node in enumerate(unscanned):
            node.set_proven_action_values(
                target_q[index],
                target_mask[index],
            )
            node.tactical_proof_scanned = True

    @staticmethod
    def _backup(path: list[tuple[_Node, int]], leaf_value: float) -> None:
        for parent, action in reversed(path):
            leaf_value = -leaf_value
            parent.visit_counts[action] += 1
            parent.total_values[action] += leaf_value

    @staticmethod
    def _commit_virtual_backup(
        path: list[tuple[_Node, int]],
        leaf_value: float,
    ) -> None:
        for parent, action in reversed(path):
            parent.virtual_visit_counts[action] -= 1
            leaf_value = -leaf_value
            parent.visit_counts[action] += 1
            parent.total_values[action] += leaf_value

    @staticmethod
    def _propagate_proof(
        path: list[tuple[_Node, int]],
        leaf_value: float,
    ) -> None:
        if leaf_value not in {-1.0, 1.0}:
            return
        solved_value = leaf_value
        for parent, action in reversed(path):
            parent.set_proven_action(action, -solved_value)
            parent_value = parent.proven_state_value
            if parent_value is None:
                break
            solved_value = parent_value


def select_action(result: SearchResult, rng: np.random.Generator) -> int:
    mask = np.asarray(result.legal_mask, dtype=bool)
    proved = np.asarray(result.proven_mask, dtype=bool)
    proof_q = np.asarray(result.proven_q_values, dtype=np.float32)
    wins = mask & proved & (proof_q == 1.0)
    non_losses = mask & ~(proved & (proof_q == -1.0))
    if bool(wins.any()):
        mask = wins
    elif bool(non_losses.any()):
        mask = non_losses

    eligible_actions = np.flatnonzero(mask)
    searched_actions = eligible_actions[result.searched_mask[eligible_actions]]
    if searched_actions.size == 0:
        raise RuntimeError("Search produced no visited legal action")
    counts = result.visit_counts[searched_actions]
    best_actions = searched_actions[counts == counts.max()]
    return int(rng.choice(best_actions))
