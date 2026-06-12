"""Human-readable strategy tables and JSON export.

Both views are organized the same way: for each decision point in the tree
(identified by its action history), a table of the acting player's combos and
their average-strategy action frequencies, sorted strongest hand first.
"""

import json

from cfr import CFRTrainer, combo_name
from evaluator import hand_class, hand_strength
from game_tree import PLAYER_NAMES, decision_nodes


def _history_label(history: tuple[str, ...]) -> str:
    return " > ".join(history) if history else "root (first to act)"


def print_strategy_tables(trainer: CFRTrainer):
    """Print one frequency table per decision point in the tree."""
    board = trainer.board
    print(f"\nBoard: {' '.join(board)}\n")

    for node in decision_nodes(trainer.root):
        player = node.player
        rows = []
        unreached = 0
        for combo in trainer.ranges[player]:
            strategy = trainer.average_strategy_at(player, combo, node)
            if strategy is None:
                unreached += 1
                continue
            rows.append((combo, strategy))

        if not rows:
            continue  # nobody ever gets here; nothing to show

        rows.sort(key=lambda r: hand_strength(list(r[0]), board), reverse=True)

        print(f"=== {PLAYER_NAMES[player]} @ {_history_label(node.history)} ===")
        header = f"{'combo':<6} {'hand':<16}" + "".join(f"{a:>10}" for a in node.actions)
        print(header)
        for combo, strategy in rows:
            cells = "".join(f"{freq:>9.1%} " for freq in strategy)
            print(f"{combo_name(combo):<6} {hand_class(list(combo), board):<16}{cells}")
        if unreached:
            print(f"({unreached} combos never reach this node)")
        print()


def strategy_dict(trainer: CFRTrainer, pot: float) -> dict:
    """Build the solved-strategy payload as a plain dict (JSON-ready).

    Shape:
      strategies[player][history][combo] = {
          strength, hand_class, actions: {action: frequency}, reached
      }
    `history` is the action sequence joined with '/', or 'root'. Combos that
    never reach a node are included with a uniform default and reached=false,
    so consumers can decide whether to show them.
    """
    board = trainer.board
    ev = trainer.expected_game_value()

    strategies = {name: {} for name in PLAYER_NAMES.values()}
    for node in decision_nodes(trainer.root):
        player = node.player
        history_key = "/".join(node.history) if node.history else "root"
        table = {}
        for combo in trainer.ranges[player]:
            strategy = trainer.average_strategy_at(player, combo, node)
            reached = strategy is not None
            if not reached:
                strategy = [1.0 / len(node.actions)] * len(node.actions)
            table[combo_name(combo)] = {
                "strength": hand_strength(list(combo), board),
                "hand_class": hand_class(list(combo), board),
                "reached": reached,
                "actions": {a: round(f, 4) for a, f in zip(node.actions, strategy)},
            }
        strategies[PLAYER_NAMES[player]][history_key] = table

    return {
        "board": list(board),
        "pot": pot,
        "iterations": trainer.iterations_run,
        "expected_pot_share": {"OOP": round(ev[0], 2), "IP": round(ev[1], 2)},
        "ranges": {
            name: [combo_name(c) for c in trainer.ranges[player]]
            for player, name in PLAYER_NAMES.items()
        },
        "strategies": strategies,
    }


def export_json(trainer: CFRTrainer, pot: float, path: str):
    """Write a single solved scenario to a JSON file (see strategy_dict)."""
    with open(path, "w") as f:
        json.dump(strategy_dict(trainer, pot), f, indent=2)
