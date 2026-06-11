"""Hand evaluation for the river solver.

This module wraps the `treys` library so the rest of the solver never has to
think about treys' internals. The public API works with plain card strings
like "Ks" or "7h" (rank + suit, suits are c/d/h/s).

treys quirk worth knowing: its evaluator returns a rank from 1 (royal flush)
to 7462 (worst possible high card), i.e. LOWER is better. That is easy to get
backwards, so we invert it here once and expose a strength where HIGHER is
better. Everything downstream can then just compare with > and <.
"""

from treys import Card, Evaluator

# treys' worst possible hand rank. Used to flip "lower is better" into
# "higher is better": strength = WORST_TREYS_RANK + 1 - treys_rank,
# giving a range of 1 (worst high card) to 7462 (royal flush).
WORST_TREYS_RANK = 7462

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
