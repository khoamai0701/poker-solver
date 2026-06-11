"""Vanilla counterfactual regret minimization (CFR) for the river game.

The big picture
---------------
CFR is self-play: both players start with arbitrary (uniform) strategies and
repeatedly play every possible deal against each other. After each pass, every
decision point measures how much better each alternative action WOULD have
done compared to the current mixed strategy ("regret"), and accumulates it.
The next strategy plays each action in proportion to its accumulated positive
regret ("regret matching"). The strategies played along the way are also
accumulated, and it is their weighted AVERAGE -- not the final iteration --
that converges to a Nash equilibrium in two-player zero-sum games.

Information sets
----------------
A player at a decision point knows their own hole cards and the betting so
far, but not the opponent's cards. So an information set is keyed by
(player, own combo, action history), and every deal that shares those three
things shares the same regrets and strategy.

This is the textbook "vanilla" algorithm: every iteration walks the full tree
for every possible deal (every non-conflicting pair of combos from the two
ranges). No sampling, no pruning -- slower, but easy to follow.
"""

from dataclasses import dataclass

from evaluator import hand_strength
from game_tree import IP, OOP, Decision, Terminal

# A combo is a tuple of two card strings, e.g. ("As", "Ad").
Combo = tuple[str, str]


def combo_name(combo: Combo) -> str:
    """Canonical display name for a combo, e.g. 'AsAd'."""
    return "".join(combo)


@dataclass
class Deal:
    """One way the river could be dealt: a combo for each player.

    `showdown_winner` is precomputed because the board is fixed: OOP, IP,
    or None for a chop.
    """

    oop: Combo
    ip: Combo
    showdown_winner: int | None


class InfoSet:
    """Accumulated regrets and strategy weight for one information set."""

    def __init__(self, num_actions: int):
        # regret_sum[a]: total counterfactual regret for not always taking a.
        self.regret_sum = [0.0] * num_actions
        # strategy_sum[a]: reach-weighted sum of the probability we played a,
        # across all iterations. Normalizing this gives the average strategy.
        self.strategy_sum = [0.0] * num_actions

    def current_strategy(self) -> list[float]:
        """Regret matching: play each action proportional to its positive regret.

        If nothing has positive regret yet (e.g. the first iteration), fall
        back to uniform.
        """
        positive = [max(r, 0.0) for r in self.regret_sum]
        total = sum(positive)
        if total > 0:
            return [p / total for p in positive]
        n = len(positive)
        return [1.0 / n] * n

    def average_strategy(self) -> list[float]:
        """The strategy that converges to Nash: normalized strategy_sum."""
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        n = len(self.strategy_sum)
        return [1.0 / n] * n


class CFRTrainer:
    def __init__(self, root: Decision, board: list[str], oop_range: list[Combo], ip_range: list[Combo]):
        self.root = root
        self.board = board
        self.ranges = {OOP: oop_range, IP: ip_range}
        # (player, combo, history) -> InfoSet, created lazily on first visit.
        self.infosets: dict[tuple, InfoSet] = {}
        self.deals = self._enumerate_deals()
        self.iterations_run = 0

    def _enumerate_deals(self) -> list[Deal]:
        """Every non-conflicting (OOP combo, IP combo) pair, with the
        showdown result precomputed since the board never changes."""
        strength = {
            combo: hand_strength(list(combo), self.board)
            for rng in self.ranges.values()
            for combo in rng
        }
        deals = []
        for oop_combo in self.ranges[OOP]:
            for ip_combo in self.ranges[IP]:
                if set(oop_combo) & set(ip_combo):
                    continue  # card removal: the two players can't share a card
                diff = strength[oop_combo] - strength[ip_combo]
                winner = OOP if diff > 0 else IP if diff < 0 else None
                deals.append(Deal(oop_combo, ip_combo, winner))
        return deals

    def train(self, iterations: int, progress_every: int = 0):
        """Run vanilla CFR. Each iteration traverses the full tree for every deal."""
        for t in range(1, iterations + 1):
            for deal in self.deals:
                # Both players start with reach probability 1: every deal in
                # the enumeration is equally likely (uniform range weights).
                self._cfr(self.root, deal, reach_oop=1.0, reach_ip=1.0)
            self.iterations_run += 1
            if progress_every and t % progress_every == 0:
                print(f"  iteration {t}/{iterations}")

    def _cfr(self, node, deal: Deal, reach_oop: float, reach_ip: float) -> tuple[float, float]:
        """Recursively evaluate `node` for one deal, updating regrets and the
        average-strategy accumulator along the way.

        reach_oop / reach_ip: the probability that each player's OWN strategy
        plays to this node (their contribution to reaching it). Returns the
        expected payoff vector (OOP, IP) under the current strategies.
        """
        if isinstance(node, Terminal):
            return node.payoffs(deal.showdown_winner)

        player = node.player
        my_combo = deal.oop if player == OOP else deal.ip
        info = self._infoset(player, my_combo, node)
        strategy = info.current_strategy()

        # Recurse into each action, scaling the acting player's reach by the
        # probability of taking that action.
        action_utils = []
        node_util = [0.0, 0.0]
        for prob, action in zip(strategy, node.actions):
            child = node.children[action]
            if player == OOP:
                util = self._cfr(child, deal, reach_oop * prob, reach_ip)
            else:
                util = self._cfr(child, deal, reach_oop, reach_ip * prob)
            action_utils.append(util)
            node_util[0] += prob * util[0]
            node_util[1] += prob * util[1]

        # Counterfactual regret: how much better action `a` does than the
        # current mix, weighted by the OPPONENT's probability of reaching this
        # node (the "counterfactual" part: we pretend the acting player always
        # got here, so their own reach is excluded from the weight).
        my_reach = reach_oop if player == OOP else reach_ip
        opp_reach = reach_ip if player == OOP else reach_oop
        for a, util in enumerate(action_utils):
            info.regret_sum[a] += opp_reach * (util[player] - node_util[player])
            # The average strategy weights each iteration's strategy by the
            # player's own reach: combos that get here often count for more.
            info.strategy_sum[a] += my_reach * strategy[a]

        return (node_util[0], node_util[1])

    def _infoset(self, player: int, combo: Combo, node: Decision) -> InfoSet:
        key = (player, combo, node.history)
        info = self.infosets.get(key)
        if info is None:
            info = InfoSet(len(node.actions))
            self.infosets[key] = info
        return info

    # ---- Inspection helpers (used by output/tests, not by training) ----

    def average_strategy_at(self, player: int, combo: Combo, node: Decision) -> list[float] | None:
        """Average strategy for a combo at a node, or None if the combo never
        reaches this node with positive probability (nothing to average)."""
        info = self.infosets.get((player, combo, node.history))
        if info is None or sum(info.strategy_sum) <= 0:
            return None
        return info.average_strategy()

    def expected_game_value(self) -> tuple[float, float]:
        """Expected payoff (OOP, IP) when both play their average strategies.

        Since payoffs are net of river investment, the two values sum to the
        pot: this is each player's share of the money in the middle.
        """
        total = [0.0, 0.0]
        for deal in self.deals:
            util = self._ev(self.root, deal)
            total[0] += util[0]
            total[1] += util[1]
        n = len(self.deals)
        return (total[0] / n, total[1] / n)

    def _ev(self, node, deal: Deal) -> tuple[float, float]:
        if isinstance(node, Terminal):
            return node.payoffs(deal.showdown_winner)
        my_combo = deal.oop if node.player == OOP else deal.ip
        info = self.infosets.get((node.player, my_combo, node.history))
        strategy = info.average_strategy() if info else None
        if strategy is None:
            strategy = [1.0 / len(node.actions)] * len(node.actions)
        util = [0.0, 0.0]
        for prob, action in zip(strategy, node.actions):
            child_util = self._ev(node.children[action], deal)
            util[0] += prob * child_util[0]
            util[1] += prob * child_util[1]
        return (util[0], util[1])
