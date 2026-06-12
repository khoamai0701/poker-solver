"""Precompute solved strategies for FELT's river lesson spots.

Defines a list of lesson Scenarios, solves each one with CFR, and writes
everything to a single lesson_strategies.json that the React app can import
statically (no backend, no live solving).

To add a spot, append a Scenario to SPOTS -- the id, solving, and export are
all derived automatically. Keep ranges around 15-30 combos per side so each
spot solves in a few seconds.

Usage:
    python precompute.py [--iterations 500] [--out lesson_strategies.json]
"""

import argparse
import json
import re
import time

from output import strategy_dict
from scenario import DEFAULT_SCENARIO, Scenario, expand_range
from solver import solve

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# The lesson spots. Each teaches one river concept; the description is shown
# to the student in FELT. Boards are 5 cards; ranges use the shorthand from
# scenario.py ("AA", "AKs", "AKo", "AK", or explicit combos like "AsQs").
# ---------------------------------------------------------------------------
SPOTS = [
    DEFAULT_SCENARIO,  # King-high dry: value vs bluff-catchers
    Scenario(
        name="Polarized vs bluff-catcher",
        board=("Ks", "7h", "2d", "8c", "3s"),
        oop_range=("KK", "QJs"),  # the nuts plus pure air
        ip_range=("99", "TT"),  # bluff-catchers only
        bet_fractions=(1.0,),
        description="The classic toy game: with one pot-sized bet, value bets "
        "and bluffs must come in a 2:1 ratio and the defender calls half the "
        "time. Watch the frequencies match the theory.",
    ),
    Scenario(
        name="Ace-high dry: top pair battles",
        board=("Ah", "Kd", "7c", "4s", "2h"),
        oop_range=("AQs", "AJs", "ATs", "KQs", "77", "44"),
        ip_range=("KK", "22", "A9s", "A8s", "QQ", "JJ", "KJs"),
        description="Both players hold top pairs of different strength, but "
        "IP's range hides sets of kings and deuces. Kicker quality decides "
        "who bets thin and who bluff-catches.",
    ),
    Scenario(
        name="Paired board: pressure on bluff-catchers",
        board=("Qs", "Qh", "8d", "5c", "2s"),
        oop_range=("AQ", "KK", "88", "AKs"),
        ip_range=("QJs", "Q9s", "55", "TT", "99", "KJs", "JTs"),
        description="Trips and overpairs attack the paired board, but IP "
        "slow-played some queens too. Middle pairs face tough calls.",
    ),
    Scenario(
        name="Flush board: nut advantage",
        board=("Js", "8s", "4s", "Kh", "2c"),
        oop_range=("AsQs", "QsTs", "9s7s", "KK", "AJs", "99"),
        ip_range=("Ts9s", "7s6s", "6s5s", "KQs", "JTs", "88"),
        description="Three spades on board. Who holds the bigger flushes "
        "dictates who can bet big and who must check-call.",
    ),
    Scenario(
        name="Straight completes on the river",
        board=("Th", "9c", "8d", "3s", "2c"),
        oop_range=("QJs", "TT", "99", "77", "AA"),
        ip_range=("J7s", "76s", "T9s", "KQs", "JJ", "A8s"),
        description="The ten-nine-eight runout puts straights in both ranges. "
        "Sets and overpairs are downgraded to bluff-catchers.",
    ),
    Scenario(
        name="Missed flush draws: who bluffs?",
        board=("Kh", "Th", "4d", "8s", "2c"),
        oop_range=("AK", "KQs", "QhJh", "Jh9h", "Ah5h"),
        ip_range=("TT", "88", "KJs", "T9s", "A4s", "66"),
        description="The river bricks the heart draw. Busted draws make the "
        "cheapest bluffs because they never win at showdown.",
    ),
    Scenario(
        name="Low board: overpairs out of position",
        board=("Jc", "9d", "5h", "3c", "2s"),
        oop_range=("QQ", "TT", "88", "AJs"),
        ip_range=("KJs", "JTs", "99", "T8s", "A5s", "KQs"),
        description="Overpairs and second pairs navigate a low, disconnected "
        "board where thin value bets go a long way.",
    ),
    Scenario(
        name="Thin value with top pair",
        board=("Qd", "9s", "6h", "3d", "2c"),
        oop_range=("KQs", "QJs", "AQs", "99", "66", "AA"),
        ip_range=("QTs", "96s", "33", "JJ", "TT", "KJs", "A3s"),
        description="How thin can top pair bet for value when worse pairs "
        "make up most of the calling range?",
    ),
    Scenario(
        name="Board pairs on the river",
        board=("As", "8c", "5d", "8h", "2s"),
        oop_range=("AKs", "AQs", "55", "88", "AA"),
        ip_range=("98s", "T8s", "A8s", "TT", "99", "A7s", "KQs"),
        description="The river pairs the eight, turning some two pairs into "
        "trips-plus and freezing medium-strength hands.",
    ),
    Scenario(
        name="Three-bet pot: overpairs vs sets",
        board=("7d", "5c", "3h", "Jd", "2s"),
        oop_range=("AA", "KK", "QQ", "AKs"),
        ip_range=("77", "55", "JJ", "TT", "AQs", "65s"),
        pot=200.0,
        description="A preflop three-bettor's overpair range meets a caller "
        "whose range hides flopped sets on a ragged board.",
    ),
    Scenario(
        name="Overbet with the nut advantage",
        board=("Ac", "Th", "6d", "5s", "2h"),
        oop_range=("AKs", "AQs", "66", "55", "QJs", "JTs"),
        ip_range=("ATs", "KTs", "99", "88", "76s", "KQs"),
        bet_fractions=(1.0, 2.0),
        description="Only the OOP range contains aces-up and sets, unlocking "
        "an overbet sizing that maximizes pressure on capped hands.",
    ),
    Scenario(
        name="Small bet on a static board",
        board=("8d", "6c", "2s", "2d", "9h"),
        oop_range=("99", "88", "66", "A8s", "KQs", "AJs"),
        ip_range=("T8s", "87s", "55", "JTs", "A6s", "K9s"),
        bet_fractions=(0.33, 1.0),
        description="On a board where equities barely move, a one-third pot "
        "bet attacks weak pairs cheaply and high frequency.",
    ),
    Scenario(
        name="Two pair vs straight threats",
        board=("Jh", "Td", "4c", "8s", "3d"),
        oop_range=("JTs", "Q9s", "AJs", "TT", "44", "AA"),
        ip_range=("97s", "KQs", "99", "A4s", "J9s"),
        description="Two pair and sets want value, but queen-nine and "
        "nine-seven straights lurk in both ranges.",
    ),
    Scenario(
        name="Monotone flop, brick runout",
        board=("Qh", "9h", "5h", "Ad", "2c"),
        oop_range=("KhJh", "Th8h", "AcQc", "AsQs", "AA", "KQs"),
        ip_range=("7h6h", "AhJc", "QJs", "99", "55", "JTs"),
        description="Flushes made on the flop slow-play or build the pot; "
        "bare top pairs decide how much heat they can stand.",
    ),
]


def spot_id(name: str) -> str:
    """Stable kebab-case id derived from the spot name, e.g.
    'Flush board: nut advantage' -> 'flush-board-nut-advantage'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def precompute(iterations: int, out_path: str):
    ids = [spot_id(s.name) for s in SPOTS]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate spot ids; give every Scenario a unique name")

    entries = []
    total_start = time.perf_counter()
    for scenario, sid in zip(SPOTS, ids):
        n_oop = len(expand_range(scenario.oop_range, scenario.board))
        n_ip = len(expand_range(scenario.ip_range, scenario.board))
        print(f"[{len(entries) + 1:>2}/{len(SPOTS)}] {sid} ({n_oop}v{n_ip} combos)...", end="", flush=True)

        start = time.perf_counter()
        trainer = solve(scenario, iterations)
        ev = trainer.expected_game_value()
        print(f" {time.perf_counter() - start:.1f}s  (OOP pot share {ev[0]:.1f})")

        # strategy_dict provides board/pot/ranges/strategies; we add the
        # lesson metadata FELT needs on top.
        entry = {
            "id": sid,
            "name": scenario.name,
            "description": scenario.description,
            "bet_fractions": list(scenario.bet_fractions),
            "raise_fractions": list(scenario.raise_fractions),
            **strategy_dict(trainer, scenario.pot),
        }
        entries.append(entry)

    data = {
        "schema_version": SCHEMA_VERSION,
        "iterations": iterations,
        "spot_count": len(entries),
        "spots": entries,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    elapsed = time.perf_counter() - total_start
    print(f"\nWrote {len(entries)} spots to {out_path} in {elapsed:.0f}s total")


def main():
    parser = argparse.ArgumentParser(description="Precompute FELT lesson strategies")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--out", default="lesson_strategies.json")
    args = parser.parse_args()
    precompute(args.iterations, args.out)


if __name__ == "__main__":
    main()
