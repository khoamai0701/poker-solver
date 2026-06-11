"""Sanity tests for the evaluator wrapper.

These mostly guard against the "lower is better" / "higher is better"
confusion: every test asserts the stronger poker hand gets the HIGHER number.
"""

import pytest

from evaluator import hand_class, hand_strength

# The fixed board used throughout the solver's test scenario.
BOARD = ["Ks", "7h", "2d", "8c", "3s"]


def test_better_hand_has_higher_strength():
    set_of_sevens = hand_strength(["7c", "7d"], BOARD)
    top_pair = hand_strength(["Ah", "Kh"], BOARD)
    ace_high = hand_strength(["Ah", "Qc"], BOARD)
    assert set_of_sevens > top_pair > ace_high


def test_hand_ordering_across_categories():
    # Strictly increasing hand categories on this board.
    hands = [
        ["Qc", "Jc"],  # queen high
        ["Ah", "2c"],  # pair of twos
        ["Ah", "Kh"],  # top pair
        ["8d", "7s"],  # two pair
        ["7c", "7d"],  # set of sevens
        ["Kh", "Kd"],  # top set
    ]
    strengths = [hand_strength(h, BOARD) for h in hands]
    assert strengths == sorted(strengths), "hands should be in increasing strength"


def test_chopped_pot_gives_equal_strength():
    # Both players play the board's king with the same kicker situation:
    # identical hand ranks -> equal strength -> chop.
    a = hand_strength(["Ah", "Kh"], BOARD)
    b = hand_strength(["Ad", "Kd"], BOARD)
    assert a == b


def test_kicker_breaks_ties():
    ace_kicker = hand_strength(["Ah", "Kh"], BOARD)
    queen_kicker = hand_strength(["Qh", "Kh"], BOARD)
    assert ace_kicker > queen_kicker


def test_hand_class_names():
    assert hand_class(["7c", "7d"], BOARD) == "Three of a Kind"
    assert hand_class(["Ah", "Kh"], BOARD) == "Pair"
    assert hand_class(["Qc", "Jc"], BOARD) == "High Card"


def test_input_validation():
    with pytest.raises(ValueError):
        hand_strength(["Ah"], BOARD)  # only one hole card
    with pytest.raises(ValueError):
        hand_strength(["Ah", "Kh"], BOARD[:4])  # 4-card board
