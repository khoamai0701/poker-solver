"""Glue between a Scenario and the CFR machinery.

`solve(scenario, iterations)` expands the ranges, builds the game tree, runs
CFR, and returns the trained CFRTrainer. Validation lives here so that a bad
spot in a big precompute batch fails loudly with a useful message instead of
producing a silently empty result.
"""

from cfr import CFRTrainer
from game_tree import build_tree
from scenario import Scenario, expand_range


def solve(scenario: Scenario, iterations: int, progress_every: int = 0) -> CFRTrainer:
    _validate_board(scenario)

    oop_range = expand_range(scenario.oop_range, scenario.board)
    ip_range = expand_range(scenario.ip_range, scenario.board)
    if not oop_range or not ip_range:
        raise ValueError(
            f"{scenario.name!r}: a range is empty after removing board conflicts "
            f"(OOP {len(oop_range)} combos, IP {len(ip_range)})"
        )

    root = build_tree(
        pot=scenario.pot,
        bet_fractions=scenario.bet_fractions,
        raise_fractions=scenario.raise_fractions,
        max_raises=scenario.max_raises,
    )
    trainer = CFRTrainer(root, list(scenario.board), oop_range, ip_range)
    if not trainer.deals:
        raise ValueError(f"{scenario.name!r}: every OOP/IP combo pair shares a card")

    trainer.train(iterations, progress_every=progress_every)
    return trainer


def _validate_board(scenario: Scenario):
    board = scenario.board
    if len(board) != 5:
        raise ValueError(f"{scenario.name!r}: board must have 5 cards, got {board}")
    if len(set(board)) != 5:
        raise ValueError(f"{scenario.name!r}: duplicate card on board {board}")
