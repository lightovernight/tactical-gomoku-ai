from __future__ import annotations

import argparse

import pygame

from .game import (
    BOARD_SIZE,
    EMPTY,
    PLAYER_BLACK,
    PLAYER_WHITE,
    GomokuPosition,
)
from .mcts import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SIMULATIONS,
    TacticalPUCT,
    select_action,
)
from .model import load_model

CELL_SIZE = 40
MARGIN = 50
BOARD_PIXELS = CELL_SIZE * (BOARD_SIZE - 1)
WINDOW_WIDTH = BOARD_PIXELS + 2 * MARGIN
WINDOW_HEIGHT = WINDOW_WIDTH + 90

BOARD_COLOR = (214, 172, 105)
GRID_COLOR = (45, 36, 25)
TEXT_COLOR = (28, 28, 28)
LAST_MOVE_COLOR = (210, 45, 45)


def mouse_to_action(mouse_position: tuple[int, int]) -> int | None:
    mouse_x, mouse_y = mouse_position
    column = round((mouse_x - MARGIN) / CELL_SIZE)
    row = round((mouse_y - MARGIN) / CELL_SIZE)
    if not (0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE):
        return None

    grid_x = MARGIN + column * CELL_SIZE
    grid_y = MARGIN + row * CELL_SIZE
    if abs(mouse_x - grid_x) > CELL_SIZE * 0.45:
        return None
    if abs(mouse_y - grid_y) > CELL_SIZE * 0.45:
        return None
    return row * BOARD_SIZE + column


def status_text(position: GomokuPosition, human_player: int) -> str:
    if position.terminated:
        if position.winner == EMPTY:
            return "Draw. Press R to play again."
        if position.winner == human_player:
            return "You win! Press R to play again."
        return "AI wins. Press R to play again."
    if position.current_player == human_player:
        return "Your turn"
    return "AI is thinking..."


def draw_game(
    screen: pygame.Surface,
    position: GomokuPosition,
    human_player: int,
    fonts: tuple[pygame.font.Font, pygame.font.Font],
    simulations: int,
) -> None:
    screen.fill(BOARD_COLOR)
    for index in range(BOARD_SIZE):
        offset = MARGIN + index * CELL_SIZE
        pygame.draw.line(
            screen,
            GRID_COLOR,
            (MARGIN, offset),
            (MARGIN + BOARD_PIXELS, offset),
            1,
        )
        pygame.draw.line(
            screen,
            GRID_COLOR,
            (offset, MARGIN),
            (offset, MARGIN + BOARD_PIXELS),
            1,
        )

    for row, column in ((3, 3), (3, 11), (7, 7), (11, 3), (11, 11)):
        pygame.draw.circle(
            screen,
            GRID_COLOR,
            (MARGIN + column * CELL_SIZE, MARGIN + row * CELL_SIZE),
            4,
        )

    board = position.board.reshape(BOARD_SIZE, BOARD_SIZE)
    stone_radius = int(CELL_SIZE * 0.43)
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            value = int(board[row, column])
            if value == EMPTY:
                continue
            center = (
                MARGIN + column * CELL_SIZE,
                MARGIN + row * CELL_SIZE,
            )
            if value == PLAYER_BLACK:
                pygame.draw.circle(screen, (18, 18, 18), center, stone_radius)
                pygame.draw.circle(screen, (55, 55, 55), center, stone_radius, 1)
            else:
                pygame.draw.circle(screen, (238, 238, 238), center, stone_radius)
                pygame.draw.circle(screen, (80, 80, 80), center, stone_radius, 1)

    if position.last_action is not None:
        row, column = divmod(position.last_action, BOARD_SIZE)
        center = (MARGIN + column * CELL_SIZE, MARGIN + row * CELL_SIZE)
        pygame.draw.circle(screen, LAST_MOVE_COLOR, center, 5, 2)

    status_font, help_font = fonts
    side = "Black" if human_player == PLAYER_BLACK else "White"
    status_surface = status_font.render(
        status_text(position, human_player),
        True,
        TEXT_COLOR,
    )
    help_surface = help_font.render(
        f"Human: {side}   MCTS: {simulations}   B/W: side   R: restart   Esc: quit",
        True,
        TEXT_COLOR,
    )
    screen.blit(status_surface, (MARGIN, WINDOW_WIDTH + 8))
    screen.blit(help_surface, (MARGIN, WINDOW_WIDTH + 43))


def run_game(
    *,
    human_player: int = PLAYER_BLACK,
    simulations: int = DEFAULT_SIMULATIONS,
    inference_batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    if human_player not in {PLAYER_BLACK, PLAYER_WHITE}:
        raise ValueError("human_player must be black or white")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")

    network, device, checkpoint = load_model()
    searcher = TacticalPUCT(
        network,
        device,
        simulations=simulations,
        inference_batch_size=inference_batch_size,
        seed=0,
    )
    print(
        "model="
        f"{checkpoint['games_completed']}-game release "
        f"device={device} simulations={simulations} "
        f"batch={inference_batch_size}",
        flush=True,
    )

    pygame.init()
    pygame.display.set_caption("Gomoku - Human vs AI")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    fonts = (
        pygame.font.SysFont("arial", 25),
        pygame.font.SysFont("arial", 17),
    )

    position = GomokuPosition.initial()
    running = True

    def restart() -> None:
        nonlocal position
        position = GomokuPosition.initial()
        searcher.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in {pygame.K_ESCAPE, pygame.K_q}:
                    running = False
                elif event.key == pygame.K_r:
                    restart()
                elif event.key == pygame.K_b:
                    human_player = PLAYER_BLACK
                    restart()
                elif event.key == pygame.K_w:
                    human_player = PLAYER_WHITE
                    restart()
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and not position.terminated
                and position.current_player == human_player
            ):
                action = mouse_to_action(event.pos)
                if action is not None and position.legal_mask[action]:
                    searcher.advance_root(action)
                    position = position.play(action)

        draw_game(screen, position, human_player, fonts, simulations)
        pygame.display.flip()

        if (
            running
            and not position.terminated
            and position.current_player != human_player
        ):
            result = searcher.run(position)
            action = select_action(result, searcher.rng)
            searcher.advance_root(action)
            position = position.play(action)

        clock.tick(60)

    pygame.quit()


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play Gomoku against the bundled 25,000-game AI"
    )
    parser.add_argument(
        "--human",
        choices=("black", "white"),
        default="black",
        help="choose your side (default: black)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help=f"MCTS simulations per AI move (default: {DEFAULT_SIMULATIONS})",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"batched leaf evaluations (default: {DEFAULT_BATCH_SIZE})",
    )
    return parser.parse_args(arguments)


def main() -> None:
    arguments = parse_args()
    human_player = PLAYER_BLACK if arguments.human == "black" else PLAYER_WHITE
    run_game(
        human_player=human_player,
        simulations=arguments.simulations,
        inference_batch_size=arguments.inference_batch_size,
    )


if __name__ == "__main__":
    main()
