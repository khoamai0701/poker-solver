"""Solve the river scenario defined in scenario.py and print/export the result.

Usage:
    python main.py [--iterations N] [--out strategy.json]
"""

import argparse
import time

import scenario
from cfr import CFRTrainer
from game_tree import IP, OOP, build_tree
from output import export_json, print_strategy_tables


def main():
    parser = argparse.ArgumentParser(description="River-only CFR solver")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--out", default="strategy.json")
    args = parser.parse_args()

    oop_range = scenario.expand_range(scenario.OOP_RANGE_SPEC, scenario.BOARD)
    ip_range = scenario.expand_range(scenario.IP_RANGE_SPEC, scenario.BOARD)

    root = build_tree(
        pot=scenario.POT,
        bet_fractions=scenario.BET_FRACTIONS,
        raise_fractions=scenario.RAISE_FRACTIONS,
        max_raises=scenario.MAX_RAISES,
    )
    trainer = CFRTrainer(root, scenario.BOARD, oop_range, ip_range)

    print(f"OOP range: {len(oop_range)} combos | IP range: {len(ip_range)} combos")
    print(f"Deals to traverse per iteration: {len(trainer.deals)}")
    print(f"Running {args.iterations} iterations of vanilla CFR...")

    start = time.perf_counter()
    trainer.train(args.iterations, progress_every=max(1, args.iterations // 10))
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.1f}s")

    print_strategy_tables(trainer)

    ev = trainer.expected_game_value()
    print(f"Expected pot share (pot = {scenario.POT:g}):")
    print(f"  OOP: {ev[OOP]:.2f}   IP: {ev[IP]:.2f}")

    export_json(trainer, scenario.POT, args.out)
    print(f"\nStrategies exported to {args.out}")


if __name__ == "__main__":
    main()
