"""Hand evaluation for the river solver.

This module wraps the `treys` library so the rest of the solver never has to
think about treys' internals. The public API works with plain card strings
like "Ks" or "7h" (rank + suit, suits are c/d/h/s).

treys quirk worth knowing: its evaluator returns a rank from 1 (royal flush)
to 7462 (worst possible high card), i.e. LOWER is better. That is easy to get
backwards, so we invert it here once and expose a strength where HIGHER is
better. Everything downstream can then just compare with > and <.
"""

from collections import Counter

from treys import Card, Evaluator

# treys' worst possible hand rank. Used to flip "lower is better" into
# "higher is better": strength = WORST_TREYS_RANK + 1 - treys_rank,
# giving a range of 1 (worst high card) to 7462 (royal flush).
WORST_TREYS_RANK = 7462

# Rank order, low to high, for board-relative comparisons (overpair vs
# underpair, top pair vs second pair). Mirrors scenario.RANKS.
_RANK_ORDER = "23456789TJQKA"


def _rank_value(rank: str) -> int:
    return _RANK_ORDER.index(rank)

# A single shared evaluator. treys builds lookup tables on construction,
# so we make one instance at import time and reuse it.
_evaluator = Evaluator()


def parse_card(card_str: str) -> int:
    """Convert a card string like 'Ks' into treys' internal int format."""
    return Card.new(card_str)


def parse_cards(card_strs: list[str]) -> list[int]:
    """Convert a list of card strings into treys ints."""
    return [parse_card(c) for c in card_strs]


def hand_strength(hole_cards: list[str], board: list[str]) -> int:
    """Return the strength of a 2-card hand on a 5-card board.

    Higher is better. Two hands can be compared directly:
        hand_strength(a, board) > hand_strength(b, board)  ->  a beats b
    Equal strengths mean a chopped pot.
    """
    if len(hole_cards) != 2:
        raise ValueError(f"expected 2 hole cards, got {hole_cards}")
    if len(board) != 5:
        raise ValueError(f"expected a 5-card board, got {board}")

    treys_rank = _evaluator.evaluate(parse_cards(board), parse_cards(hole_cards))
    return WORST_TREYS_RANK + 1 - treys_rank


def hand_class(hole_cards: list[str], board: list[str]) -> str:
    """Human-readable hand category, e.g. 'Pair' or 'Flush'. For display only."""
    treys_rank = _evaluator.evaluate(parse_cards(board), parse_cards(hole_cards))
    return _evaluator.class_to_string(_evaluator.get_rank_class(treys_rank))


def semantic_strength(hole_cards: list[str], board: list[str]) -> str:
    """Poker-semantic hand label, refining treys' rank class with board context.

    treys' hand_class only knows the made-hand category ("Pair", "Three of a
    Kind", ...); it cannot tell an overpair from an underpair, or a set from
    trips, because that depends on how the hole cards relate to the board. This
    function resolves those distinctions so a lesson can pin itself to a concept
    (e.g. always deal "top_pair" on a kicker-battle spot).

    Returns one of:
        high_card, underpair, weak_pair, second_pair, top_pair, overpair,
        two_pair, trips, set, straight, flush, full_house, quads,
        straight_flush
    """
    if len(hole_cards) != 2:
        raise ValueError(f"expected 2 hole cards, got {hole_cards}")
    if len(board) != 5:
        raise ValueError(f"expected a 5-card board, got {board}")

    cls = hand_class(hole_cards, board)

    # Categories that need no board-relative refinement.
    simple = {
        "High Card": "high_card",
        "Two Pair": "two_pair",
        "Straight": "straight",
        "Flush": "flush",
        "Full House": "full_house",
        "Four of a Kind": "quads",
        "Straight Flush": "straight_flush",
    }
    if cls in simple:
        return simple[cls]

    hole_ranks = [c[0] for c in hole_cards]
    board_ranks = [c[0] for c in board]
    # Distinct board ranks, highest first: index 0 is the top board card.
    board_by_height = sorted(set(board_ranks), key=_rank_value, reverse=True)
    is_pocket_pair = hole_ranks[0] == hole_ranks[1]

    if cls == "Three of a Kind":
        # A pocket pair that matched one board card is a set; one hole card
        # filling a board pair (or a board that already shows trips) is trips.
        return "set" if is_pocket_pair else "trips"

    if cls == "Pair":
        if is_pocket_pair:
            # Pocket pair with no board card of its rank: over- or underpair,
            # decided against the highest board card.
            pair_rank = hole_ranks[0]
            top_board = board_by_height[0]
            return "overpair" if _rank_value(pair_rank) > _rank_value(top_board) else "underpair"

        # One hole card paired a board card (or both hole cards missed and the
        # pair sits on the board). Find the rank that occurs exactly twice.
        counts = Counter(hole_ranks + board_ranks)
        paired = [r for r, n in counts.items() if n == 2]
        pair_rank = paired[0] if paired else None
        if pair_rank is None or pair_rank not in hole_ranks:
            # The pair belongs to the board; the hole cards add nothing -- this
            # is effectively air for betting purposes.
            return "high_card"
        position = board_by_height.index(pair_rank)
        if position == 0:
            return "top_pair"
        if position == 1:
            return "second_pair"
        return "weak_pair"

    # Defensive default; every treys class above is handled.
    return "high_card"


if __name__ == "__main__":
    # Quick demo on the board we'll use for the solver's test scenario.
    board = ["Ks", "7h", "2d", "8c", "3s"]
    hands = [
        ["As", "Ad"],  # overpair
        ["Ah", "Kh"],  # top pair, top kicker
        ["7c", "7d"],  # set of sevens
        ["Qc", "Jc"],  # queen high (air)
        ["8d", "7s"],  # two pair
    ]

    print(f"Board: {' '.join(board)}\n")
    for hand in sorted(hands, key=lambda h: hand_strength(h, board), reverse=True):
        strength = hand_strength(hand, board)
        print(f"{hand[0]} {hand[1]}  strength={strength:>4}  {hand_class(hand, board)}")
