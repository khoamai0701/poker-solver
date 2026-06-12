"""Scenario definition: everything that describes one river spot.

A Scenario bundles the inputs the solver needs -- board, ranges, pot, and
bet-sizing options -- so spots can be defined as data and solved in bulk
(see precompute.py). The original hardcoded test spot lives on here as
DEFAULT_SCENARIO, used by main.py.

Range shorthand
---------------
Ranges are lists of specs, expanded to explicit combos with `expand_range`:
  "AA"   -> all pocket pairs of aces
  "AKs"  -> suited ace-king only
  "AKo"  -> offsuit ace-king only
  "AK"   -> all ace-king combos
  "AsQs" -> exactly that combo (useful for curating specific flushes etc.)
Combos that use a board card are dropped automatically, and duplicates
across specs are removed.
"""

from dataclasses import dataclass

RANKS = "23456789TJQKA"  # low to high
SUITS = "cdhs"


@dataclass(frozen=True)
class Scenario:
    """One river spot to solve. Ranges are shorthand specs (see module doc)."""

    name: str
    board: tuple[str, ...]  # exactly 5 cards, e.g. ("Ks", "7h", ...)
    oop_range: tuple[str, ...]
    ip_range: tuple[str, ...]
    pot: float = 100.0
    bet_fractions: tuple[float, ...] = (0.5, 1.0)
    raise_fractions: tuple[float, ...] = (1.0,)
    max_raises: int = 1
    description: str = ""


# The spot the solver was originally built around: a tight value-heavy OOP
# range against a wider, weaker IP range on a dry king-high board.
DEFAULT_SCENARIO = Scenario(
    name="King-high dry: value vs bluff-catchers",
    board=("Ks", "7h", "2d", "8c", "3s"),
    oop_range=("AA", "KK", "AK", "88", "77"),
    ip_range=("KQs", "KJs", "99", "87s", "A7s", "QJs", "T9s"),
    pot=100.0,
    bet_fractions=(0.5, 1.0),
    description="A range-advantage spot: OOP's range is almost all strong "
    "made hands, IP holds mostly bluff-catchers.",
)


def expand_range(specs, board) -> list[tuple[str, str]]:
    """Expand shorthand specs into explicit combos, excluding combos that
    conflict with the board and deduplicating across specs."""
    blocked = set(board)
    combos = {}  # dict keyed by combo, used as an ordered set
    for spec in specs:
        for combo in _expand_one(spec):
            if blocked & set(combo):
                continue
            combos[combo] = None
    return list(combos)


def _expand_one(spec: str) -> list[tuple[str, str]]:
    # Explicit combo like "AsQs": two full cards back to back.
    if len(spec) == 4 and spec[1] in SUITS and spec[3] in SUITS:
        card1, card2 = spec[:2], spec[2:]
        if card1[0] not in RANKS or card2[0] not in RANKS or card1 == card2:
            raise ValueError(f"bad explicit combo: {spec!r}")
        return [_canonical(card1, card2)]

    rank1, rank2, modifier = spec[0], spec[1], spec[2:]
    if rank1 not in RANKS or rank2 not in RANKS or modifier not in ("", "s", "o"):
        raise ValueError(f"bad range spec: {spec!r}")

    combos = []
    if rank1 == rank2:  # pocket pair: all suit pairs
        for i, s1 in enumerate(SUITS):
            for s2 in SUITS[i + 1 :]:
                combos.append(_canonical(rank1 + s1, rank2 + s2))
    else:
        for s1 in SUITS:
            for s2 in SUITS:
                if modifier == "s" and s1 != s2:
                    continue
                if modifier == "o" and s1 == s2:
                    continue
                combos.append(_canonical(rank1 + s1, rank2 + s2))
    return combos


def _canonical(card1: str, card2: str) -> tuple[str, str]:
    """Order a combo's cards high rank first (suit breaks ties) so the same
    combo always has the same representation, e.g. ('As', 'Qs')."""
    key = lambda card: (RANKS.index(card[0]), SUITS.index(card[1]))
    return tuple(sorted((card1, card2), key=key, reverse=True))
