"""CFR convergence tests.

The main test pits a perfectly polarized OOP range (the nuts plus pure air,
50/50) against an IP range that is a single bluff catcher, with one pot-sized
bet allowed. This game has a known analytic Nash equilibrium:

  - OOP bets the nuts always and bluffs the air half the time
    (bluff:value ratio = bet / (pot + bet) = 1/2 for a pot-sized bet).
  - IP calls half the time (pot odds make IP exactly indifferent).
  - OOP's expected share of the pot is 75% (IP loses an extra pot/4 by
    holding a bluff catcher against a polarized range).

If vanilla CFR is implemented correctly, the average strategy must approach
these numbers.
"""

import pytest

from cfr import CFRTrainer
from game_tree import IP, OOP, build_tree

BOARD = ["Ks", "7h", "2d", "8c", "3s"]
POT = 100.0

NUTS = ("Kh", "Kd")  # top set: the best possible hand on this board
AIR = ("Th", "9h")  # ten-high: loses to everything
BLUFF_CATCHER = ("9c", "9d")  # beats air, loses to the nuts


@pytest.fixture(scope="module")
def solved():
    root = build_tree(pot=POT, bet_fractions=(1.0,), raise_fractions=(1.0,), max_raises=1)
    trainer = CFRTrainer(root, BOARD, oop_range=[NUTS, AIR], ip_range=[BLUFF_CATCHER])
    trainer.train(5000)
    return trainer


def _strategy(trainer, player, combo, node):
    strategy = trainer.average_strategy_at(player, combo, node)
    assert strategy is not None
    return dict(zip(node.actions, strategy))


def test_oop_always_bets_the_nuts(solved):
    strategy = _strategy(solved, OOP, NUTS, solved.root)
    assert strategy["bet 100"] > 0.97


def test_oop_bluffs_air_half_the_time(solved):
    strategy = _strategy(solved, OOP, AIR, solved.root)
    assert strategy["bet 100"] == pytest.approx(0.5, abs=0.05)


def test_ip_calls_pot_bet_half_the_time(solved):
    vs_bet = solved.root.children["bet 100"]
    strategy = _strategy(solved, IP, BLUFF_CATCHER, vs_bet)
    assert strategy["call"] == pytest.approx(0.5, abs=0.05)
    # Raising a bluff catcher into a polarized range only ever gets
    # called by the nuts; equilibrium raises (almost) never.
    assert strategy["raise 100"] < 0.02


def test_game_value_matches_theory(solved):
    ev_oop, ev_ip = solved.expected_game_value()
    assert ev_oop + ev_ip == pytest.approx(POT)
    assert ev_oop == pytest.approx(0.75 * POT, abs=2.0)


def test_average_strategies_are_distributions(solved):
    for info in solved.infosets.values():
        strategy = info.average_strategy()
        assert sum(strategy) == pytest.approx(1.0)
        assert all(p >= 0 for p in strategy)


def test_card_removal_in_deals():
    root = build_tree(pot=POT, bet_fractions=(1.0,))
    # IP's combo shares the Kd with one OOP combo: that pairing must be skipped.
    trainer = CFRTrainer(
        root, BOARD, oop_range=[("Kh", "Kd"), ("Ah", "Ad")], ip_range=[("Kd", "Qd")]
    )
    assert len(trainer.deals) == 1
    assert trainer.deals[0].oop == ("Ah", "Ad")
