"""The test scenario: a fixed board, two ranges, pot, and bet sizes.

Ranges are written in standard poker shorthand and expanded to explicit
combos, automatically dropping any combo that uses a board card:
  "AA"  -> all pocket pairs of aces
  "AKs" -> suited ace-king only
  "AKo" -> offsuit ace-king only
  "AK"  -> all ace-king combos
"""

RANKS = "23456789TJQKA"  # low to high
SUITS = "cdhs"

BOARD = ["Ks", "7h", "2d", "8c", "3s"]
POT = 100.0
BET_FRACTIONS = (0.5, 1.0)  # OOP/IP may bet 50% or 100% of pot
RAISE_FRACTIONS = (1.0,)  # one raise size: pot-sized
MAX_RAISES = 1

# OOP: tight and value-heavy -- big pairs, top pair, and the two sets.
OOP_RANGE_SPEC = ["AA", "KK", "AK", "88", "77"]

# IP: wider -- top pairs with worse kickers, middle pairs, two pair,
# and some air that missed completely.
IP_RANGE_SPEC = ["KQs", "KJs", "99", "87s", "A7s", "QJs", "T9s"]


def expand_range(specs: list[str], board: list[str]) -> list[tuple[str, str]]:
    """Expand shorthand like ['AA', 'KQs'] into explicit combos, excluding
    combos that conflict with the board."""
    blocked = set(board)
    combos = []
    for spec in specs:
        for combo in _expand_one(spec):
            if blocked & set(combo):
                continue
            combos.append(combo)
    return combos


def _expand_one(spec: str) -> list[tuple[str, str]]:
    rank1, rank2, modifier = spec[0], spec[1], spec[2:]
    if rank1 not in RANKS or rank2 not in RANKS or modifier not in ("", "s", "o"):
        raise ValueError(f"bad range spec: {spec!r}")

    combos = []
    if rank1 == rank2:  # pocket pair: all suit pairs
        for i, s1 in enumerate(SUITS):
            for s2 in SUITS[i + 1 :]:
                combos.append((rank1 + s1, rank2 + s2))
    else:
        for s1 in SUITS:
            for s2 in SUITS:
                if modifier == "s" and s1 != s2:
                    continue
                if modifier == "o" and s1 == s2:
                    continue
                combos.append((rank1 + s1, rank2 + s2))
    return combos
