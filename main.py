"""Solve a single river scenario and print/export the result.

By default this solves DEFAULT_SCENARIO from scenario.py. To solve many spots
in one go (for the FELT training app), use precompute.py instead.

Usage:
    python main.py [--iterations N] [--out strategy.json]
"""

import argparse
import time

from game_tree import IP, OOP
from output import export_json, print_strategy_tables
from scenario import DEFAULT_SCENARIO, expand_range
from solver import solve


def main():
    parser = argparse.ArgumentParser(description="River-only CFR solver")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--out", default="strategy.json")
    args = parser.parse_args()

    scenario = DEFAULT_SCENARIO
    oop_range = expand_range(scenario.oop_range, scenario.board)
    ip_range = expand_range(scenario.ip_range, scenario.board)

    print(f"Scenario: {scenario.name}")
    print(f"OOP range: {len(oop_range)} combos | IP range: {len(ip_range)} combos")
    print(f"Running {args.iterations} iterations of vanilla CFR...")

    start = time.perf_counter()
    trainer = solve(scenario, args.iterations, progress_every=max(1, args.iterations // 10))
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.1f}s")

    print_strategy_tables(trainer)

    ev = trainer.expected_game_value()
    print(f"Expected pot share (pot = {scenario.pot:g}):")
    print(f"  OOP: {ev[OOP]:.2f}   IP: {ev[IP]:.2f}")

    export_json(trainer, scenario.pot, args.out)
    print(f"\nStrategies exported to {args.out}")


if __name__ == "__main__":
    main()
