"""Game tree for heads-up river play.

The river betting structure modeled here:

  1. OOP acts first: check or bet (one branch per configured bet size).
  2. Facing a bet, a player may fold, call, or raise (raises are capped,
     default one raise total, so the tree stays small).
  3. Facing a raise, the original bettor may only fold or call.
  4. If OOP checks, IP may check back (showdown) or bet, and OOP then
     responds with fold/call/raise.

Terminal nodes are folds or showdowns. Stacks are assumed deep enough to
cover every line (no all-in logic).

Bet sizing convention:
  - A bet of fraction f is f * (current pot).
  - A raise of fraction f means raising BY f * (pot after the pending bet
    is called) -- the standard "pot-sized raise" convention. So facing a
    bet B into pot P, a fraction-f raiser puts in B + f * (P + 2B) total.

All chip amounts in terminals are tracked as "river investments": chips each
player put in during river betting, separate from the pot that existed when
the river action started. Payoffs are net of a player's own investment.
"""

from dataclasses import dataclass

# Player indices used everywhere in the solver.
OOP, IP = 0, 1
PLAYER_NAMES = {OOP: "OOP", IP: "IP"}


@dataclass(frozen=True)
class Terminal:
    """A leaf of the tree: someone folded, or we reached showdown."""

    pot: float  # pot at the start of river action (dead money)
    invested: tuple[float, float]  # river chips put in by (OOP, IP)
    folder: int | None  # player who folded, or None for a showdown

    def payoffs(self, showdown_winner: int | None) -> tuple[float, float]:
        """Net river result for (OOP, IP).

        `showdown_winner` is OOP, IP, or None for a chopped pot; it is
        ignored at fold terminals. Payoffs are net of each player's own
        river investment, so they always sum to `pot` (a constant-sum
        game, which is all CFR needs).
        """
        winner = (1 - self.folder) if self.folder is not None else showdown_winner

        if winner is None:  # chop: investments were equal, split the pot
            return (self.pot / 2, self.pot / 2)

        loser = 1 - winner
        result = [0.0, 0.0]
        result[winner] = self.pot + self.invested[loser]
        result[loser] = -self.invested[loser]
        return (result[0], result[1])


@dataclass(frozen=True)
class Decision:
    """A node where one player chooses an action."""

    player: int  # OOP or IP
    history: tuple[str, ...]  # action labels leading here, e.g. ("check", "bet 50")
    actions: tuple[str, ...]  # legal action labels, in display order
    children: dict  # action label -> Decision | Terminal


def _pct(fraction: float) -> str:
    """Format a pot fraction as a label suffix: 0.5 -> '50', 1.0 -> '100'."""
    return f"{fraction * 100:g}"


def build_tree(
    pot: float,
    bet_fractions: tuple[float, ...] = (0.5, 1.0),
    raise_fractions: tuple[float, ...] = (1.0,),
    max_raises: int = 1,
):
    """Build the river game tree. Returns the root Decision node (OOP to act)."""

    def make(player, invested, history, raises_used):
        to_call = invested[1 - player] - invested[player]
        current_pot = pot + invested[0] + invested[1]
        children = {}

        if to_call == 0:
            # No bet pending: the player may check or bet.
            if player == OOP:
                # OOP's check passes the action to IP.
                children["check"] = make(IP, invested, history + ("check",), raises_used)
            else:
                # IP checking back closes the action: showdown.
                children["check"] = Terminal(pot, invested, folder=None)

            for f in bet_fractions:
                label = f"bet {_pct(f)}"
                bet = f * current_pot
                new_invested = _add(invested, player, bet)
                children[label] = make(1 - player, new_invested, history + (label,), raises_used)
        else:
            # Facing a bet or raise: fold, call, or (if not capped) raise.
            children["fold"] = Terminal(pot, invested, folder=player)
            children["call"] = Terminal(pot, _add(invested, player, to_call), folder=None)

            if raises_used < max_raises:
                pot_after_call = current_pot + to_call
                for f in raise_fractions:
                    label = f"raise {_pct(f)}"
                    raise_to = to_call + f * pot_after_call
                    new_invested = _add(invested, player, raise_to)
                    children[label] = make(
                        1 - player, new_invested, history + (label,), raises_used + 1
                    )

        return Decision(player, history, tuple(children), children)

    return make(OOP, (0.0, 0.0), (), 0)


def _add(invested: tuple[float, float], player: int, amount: float) -> tuple[float, float]:
    """Return a new invested tuple with `amount` added for `player`."""
    new = list(invested)
    new[player] += amount
    return (new[0], new[1])


def decision_nodes(node) -> list[Decision]:
    """All Decision nodes in depth-first order (used for display/export)."""
    if isinstance(node, Terminal):
        return []
    found = [node]
    for action in node.actions:
        found.extend(decision_nodes(node.children[action]))
    return found
