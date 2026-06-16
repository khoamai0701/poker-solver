"""Precompute solved strategies for FELT's river lesson spots.

Defines a curated list of lesson Scenarios, solves each one with CFR, and
writes everything to a single lesson_strategies.json that the React app can
import statically (no backend, no live solving).

Curation philosophy
--------------------
This is a hand-picked ~35-spot set chosen for instructional value and
realistic play, not a large auto-generated dump. Two layers keep it clean:

  1. Board-quality filters (board_quality_violations) reject boards that are
     poker-awkward to teach on: boards that already make a straight, boards
     with four-or-more to a flush, and boards so coordinated (four-to-a-
     straight) that a normal made hand would not want to bet/raise. These run
     before solving and fail loudly, so a bad board can never sneak in.

  2. The EV degeneracy guard (is_degenerate) drops any spot where one range so
     dominates that the weaker side's pot share is ~zero -- the "always fold"
     non-lesson.

Each Scenario is also built so the hero hand and the bet line are coherent:
the strongest part of the betting range is a hand a competent player would
actually take the defined line with (top set / nut flush / two pair value
bets, overpairs bluff-catch, etc.). The hero hand and line are spelled out in
each description so FELT can show them and they survive into the JSON.

To add a spot, append a Scenario to the relevant category list -- the id,
board validation, solving, and export are all derived automatically. Keep
ranges around 12-26 combos per side so each spot solves in a couple of seconds.

Usage:
    python precompute.py [--iterations 300] [--out lesson_strategies.json]
"""

import argparse
import json
import re
import time
from collections import Counter

from output import strategy_dict
from scenario import RANKS, Scenario, expand_range
from solver import solve

SCHEMA_VERSION = 1

# A spot is dropped if the weaker player's expected pot share falls below this
# fraction of the pot. Below ~3% the dominated range is folding almost always,
# so there is no genuine decision left to teach.
MIN_POT_SHARE_FRACTION = 0.03


# ---------------------------------------------------------------------------
# Board-quality filters
# ---------------------------------------------------------------------------
def _suit_counts(board) -> Counter:
    return Counter(card[1] for card in board)


def board_four_flush(board) -> bool:
    """True if any suit appears 4+ times -- anyone with one card of the suit
    has a flush, which makes for a degenerate, hard-to-teach river."""
    return max(_suit_counts(board).values()) >= 4


def board_straightiness(board) -> int:
    """Largest number of distinct board ranks that fall inside any window of
    five consecutive ranks (the ace is counted both high and low).

      5 -> the board itself makes a straight
      4 -> four-to-a-straight (so coordinated that made hands hate betting)
    <=3 -> a normal, teachable texture
    """
    idx = {RANKS.index(card[0]) for card in board}
    if RANKS.index("A") in idx:
        idx.add(-1)  # ace plays low for wheel straights
    best = 0
    for low in range(-1, len(RANKS)):
        best = max(best, sum(1 for i in idx if low <= i < low + 5))
    return best


def board_quality_violations(board) -> list[str]:
    """Return the reasons a board is unsuitable for a clean lesson, or []."""
    violations = []
    if board_four_flush(board):
        violations.append("four-or-more to a flush")
    straightiness = board_straightiness(board)
    if straightiness >= 5:
        violations.append("board already makes a straight")
    elif straightiness >= 4:
        violations.append("four-to-a-straight (too coordinated)")
    return violations


# ---------------------------------------------------------------------------
# The curated lesson spots, grouped by texture / concept. Boards are 5 cards;
# ranges use the shorthand from scenario.py ("AA", "AKs", "AKo", "AK", or
# explicit combos like "AsQs"). Each description names the hero hand and the
# line so the lesson is self-explanatory in FELT.
# ---------------------------------------------------------------------------

# --- Dry, disconnected boards (single-raised pots). -------------------------
DRY = [
    Scenario(
        name="King-high dry: range advantage",
        board=("Ks", "7h", "2d", "8c", "3s"),
        oop_range=("AA", "KK", "AK", "88", "77"),
        ip_range=("KQs", "KJs", "99", "87s", "A7s", "QJs", "T9s"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) holds top set and bets 1/2 to pot for value "
        "into a capped bluff-catching range. The lesson: how a clean range "
        "advantage on a dry board lets you bet often and large.",
    ),
    Scenario(
        name="Ace-high dry: top-pair kicker battle",
        board=("Ah", "Kd", "7c", "4s", "2h"),
        oop_range=("AQs", "AJs", "ATs", "KQs", "77", "44"),
        ip_range=("KK", "22", "A9s", "A8s", "QQ", "JJ", "KJs"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets top pair good kicker (AQ) on a dry "
        "ace-high board, while IP hides sets. Kicker quality decides who bets "
        "thin and who bluff-catches.",
    ),
    Scenario(
        name="Queen-high dry: two-pair value",
        board=("Qd", "7s", "3h", "Jc", "2d"),
        oop_range=("QJs", "AQs", "JJ", "33", "Kh9h", "T9s"),
        ip_range=("QTs", "Q9s", "77", "AJs", "KJs", "A5s"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) bets queen-jack two pair for value on a "
        "disconnected high-card board; bluff-catching queens have to decide "
        "how light to call.",
    ),
    Scenario(
        name="King-queen broadway dry",
        board=("Kd", "Qc", "5h", "2s", "7d"),
        oop_range=("KQs", "AKs", "55", "AhJh", "Th9h", "QJs"),
        ip_range=("KJs", "KTs", "QJs", "77", "A5s", "JhTh"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets king-queen two pair on a dry "
        "two-broadway board; missed straight draws make the natural bluffs.",
    ),
    Scenario(
        name="Jack-high dry: thin value",
        board=("Jd", "6c", "2h", "9s", "4d"),
        oop_range=("JTs", "AJs", "99", "44", "KhQh", "Ah5h"),
        ip_range=("J9s", "QJs", "66", "88", "T9s", "A6s"),
        bet_fractions=(0.5,),
        description="Hero (OOP) makes a thin half-pot value bet with top pair "
        "(AJ) on a low jack-high board where worse pairs still call.",
    ),
    Scenario(
        name="Ten-high dry: top set value",
        board=("Td", "5c", "2h", "7s", "3d"),
        oop_range=("TT", "55", "ATs", "KhQh", "Ah9h", "QJs"),
        ip_range=("T9s", "T8s", "77", "33", "A5s", "JhQh"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) bets top set (TT) or a lower set (55) on a "
        "ten-high board where every value bet is thin and overcards make the bluffs.",
    ),
    Scenario(
        name="Ace-high dry: weak-ace defense",
        board=("As", "8d", "3c", "9h", "2s"),
        oop_range=("A7s", "A5s", "99", "33", "KQs", "QJs"),
        ip_range=("ATs", "A9s", "88", "KhJh", "QhTh", "77"),
        bet_fractions=(0.5,),
        description="Hero (OOP) bets a weak ace small for thin value and to "
        "deny equity; the lesson is sizing down with a marginal made hand.",
    ),
]

# --- Paired boards. ---------------------------------------------------------
PAIRED = [
    Scenario(
        name="Paired aces: kicker war",
        board=("Ah", "Ad", "7c", "4s", "2h"),
        oop_range=("AKs", "AQs", "77", "KhQh", "JhTh", "KhJh"),
        ip_range=("AJs", "ATs", "22", "99", "A5s", "KQs"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets trip aces with the best kicker; "
        "77 makes a full house (sevens full of aces) for additional value. "
        "Weaker trips bluff-catch and busted broadways bluff. Paired-ace "
        "boards are largely a kicker contest.",
    ),
    Scenario(
        name="Paired sixes: full house and big pairs",
        board=("6s", "6d", "Kc", "3h", "2s"),
        oop_range=("KK", "AA", "66", "QhJh", "AhTh", "QQ"),
        ip_range=("K9s", "KJs", "99", "A6s", "55", "JTs"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets a full house (KK = kings full of "
        "sixes), big two pair (AA, QQ), or quads (66) on a paired low board "
        "topped by a king; middle pairs IP holds have to call or fold.",
    ),
    Scenario(
        name="Paired kings: trips vs underpair",
        board=("Ks", "Kh", "9d", "4c", "7s"),
        oop_range=("KQs", "99", "AhQh", "JhTh", "QhJh", "AhJh"),
        ip_range=("KJs", "KTs", "QQ", "A9s", "JJ", "TT"),
        bet_fractions=(0.33, 1.0),
        description="Hero (OOP) bets trip kings, picking a small or polar size; "
        "the lesson is whether IP's underpairs (QQ, JJ, TT -- all below the "
        "paired kings) are worth calling on a paired-king board.",
    ),
    Scenario(
        name="Board pairs eights: boats freeze pairs",
        board=("As", "8c", "5d", "8h", "2s"),
        oop_range=("AKs", "AQs", "55", "88", "AA"),
        ip_range=("98s", "T8s", "A8s", "TT", "99", "A7s", "KQs"),
        bet_fractions=(0.5, 1.0),
        description="The river pairs the eight. Hero (OOP) value bets a full "
        "house (AA = aces full of eights, 55 = fives full of eights) and quads "
        "while AK and AQ make two pair (aces and eights) and bluff-catch.",
    ),
    Scenario(
        name="Trips on board: kicker plays",
        board=("4c", "4d", "4h", "Qs", "9d"),
        oop_range=("AhKh", "AQs", "QQ", "KhJh", "Th8h", "99"),
        ip_range=("KQs", "Q9s", "JJ", "A5s", "TT", "JhTh"),
        bet_fractions=(0.33, 1.0),
        description="With trips on the board, full houses lead the range: QQ "
        "makes queens full of fours and 99 makes nines full of fours. Hero "
        "(OOP) value bets the best kicker (ace-high trips) or full house; "
        "weaker trips bluff-catch or give up.",
    ),
    Scenario(
        name="3-bet pot, paired queens: full house vs trips",
        board=("Qd", "Qs", "6h", "3c", "Jd"),
        oop_range=("AA", "KK", "AhQh", "JJ", "KhTh", "AdKd"),
        ip_range=("QJs", "Q9s", "TT", "A5s", "KQs", "99"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        description="In a 3-bet pot, Hero (OOP) value bets a full house "
        "(JJ = jacks full of queens) or big two pair (AA, KK paired with the "
        "board queens); the lesson is sizing against IP's slow-played trip queens. "
        "The bloated pot rewards getting value in.",
    ),
]

# --- Two-tone boards (max two of a suit -> no flush possible). ---------------
TWO_TONE = [
    Scenario(
        name="Two-tone broadway brick: two-pair value",
        board=("Ad", "Qd", "8c", "4s", "3h"),
        oop_range=("AQs", "AcKc", "QJs", "88", "KdJd", "Th9h"),
        ip_range=("AJs", "ATs", "QdTd", "44", "Jd9d", "KhJh"),
        bet_fractions=(0.75,),
        description="The diamond draw bricks. Hero (OOP) bets ace-queen two "
        "pair 3/4 pot for value while missed flush draws look for fold equity.",
    ),
    Scenario(
        name="Two-tone king board: top pair",
        board=("Kd", "Td", "7c", "4s", "2h"),
        oop_range=("AKs", "KQs", "TT", "KhQh", "JhTh", "99"),
        ip_range=("KJs", "KTs", "QJs", "A5s", "QdJd", "88"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets top pair / trip tens after the flush "
        "draw misses; IP defends second pair and weaker kings.",
    ),
    Scenario(
        name="Two-tone queen-high: set and top pair",
        board=("Qh", "9h", "5c", "3d", "2s"),
        oop_range=("AQs", "QJs", "99", "55", "KhJh", "Th8h"),
        ip_range=("QTs", "Q9s", "JJ", "A5s", "T9s", "K9s"),
        bet_fractions=(0.5,),
        description="Hero (OOP) bets a set / top pair on a queen-high two-tone "
        "board where no flush is possible; thin value against weaker queens.",
    ),
    Scenario(
        name="Two-tone ace-ten: two pair",
        board=("As", "Ts", "6d", "4c", "2h"),
        oop_range=("AQs", "AJs", "ATs", "66", "KhQh", "Jh9h"),
        ip_range=("A6s", "A4s", "TT", "KsQs", "QhJh", "T9s"),
        bet_fractions=(0.33, 0.75),
        description="Hero (OOP) value bets ace-ten two pair on a two-tone "
        "ace-high board; weaker aces and sets decide how much to pay.",
    ),
    Scenario(
        name="Two-tone jack-ten: two pair",
        board=("Jc", "Tc", "6h", "3s", "2d"),
        oop_range=("JTs", "AJs", "ATs", "KdQd", "Qh9h", "33"),
        ip_range=("J9s", "T9s", "QJs", "A5s", "KsJs", "88"),
        bet_fractions=(0.5,),
        description="Hero (OOP) bets jack-ten two pair on a connected but "
        "flush-impossible board; one-pair hands have to find the right calls.",
    ),
]

# --- Monotone / flush-possible boards (exactly three of a suit). ------------
FLUSH = [
    Scenario(
        name="Flush possible: flush vs made hands",
        board=("Qh", "9h", "5h", "Ad", "2c"),
        oop_range=("KhJh", "Th8h", "AcQc", "AsQs", "AA", "KQs"),
        ip_range=("7h6h", "AhJc", "QJs", "99", "55", "JTs"),
        bet_fractions=(0.5, 1.0),
        description="Three hearts are out. Hero (OOP) value bets a made flush "
        "while two pair (AcQc, AsQs), a set of aces (AA), and single pair "
        "(KQs) are demoted to bluff-catchers.",
    ),
    Scenario(
        name="Flush board: nut advantage",
        board=("Js", "8s", "4s", "Kh", "2c"),
        oop_range=("AsQs", "QsTs", "9s7s", "KK", "AJs", "99"),
        ip_range=("Ts9s", "7s6s", "6s5s", "KQs", "JTs", "88"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) bets the nut flush; who holds the bigger "
        "flushes dictates who bets big and who must check-call.",
    ),
    Scenario(
        name="King-high flush board",
        board=("Kh", "Th", "6h", "4s", "2c"),
        oop_range=("AhQh", "Jh9h", "KK", "TT", "AcKc", "QsJs"),
        ip_range=("Qh8h", "7h5h", "KQs", "66", "A5s", "T9s"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets the ace-high flush; top set without "
        "a heart drops to a bluff-catcher against the smaller flushes.",
    ),
    Scenario(
        name="Low flush board: nut flush vs set",
        board=("9c", "6c", "3c", "Ah", "Kd"),
        oop_range=("AcQc", "Tc8c", "99", "AsAd", "KhQh", "JhTh"),
        ip_range=("Kc7c", "6c5c", "66", "A6s", "QhJh", "T9s"),
        bet_fractions=(0.5, 1.0),
        description="Three low clubs. Hero (OOP) bets the nut / strong flush "
        "for stacks while a rivered set of aces tries to bluff-catch.",
    ),
    Scenario(
        name="Broadway flush board",
        board=("Kd", "Qd", "7d", "4s", "2c"),
        oop_range=("AdJd", "Td9d", "KQs", "QQ", "AcKc", "JhTh"),
        ip_range=("Jd8d", "9d8d", "KJs", "77", "A5s", "T9s"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets the nut flush (or king-queen two "
        "pair) on a high two-tone-to-flush board; smaller diamonds bluff-catch.",
    ),
]

# --- Broadway boards (high cards, no made straight, flush not live). --------
BROADWAY = [
    Scenario(
        name="KQJ broadway: straights and two pair",
        board=("Kh", "Qc", "Jd", "5s", "2h"),
        oop_range=("AKs", "KQs", "AhTh", "QhJh", "AA", "JJ"),
        ip_range=("AQs", "KJs", "ATs", "A5s", "TT", "QJs"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets the broadway straight (AT) and two "
        "pair; on this connected high board, sets and overpairs become "
        "bluff-catchers against the straight.",
    ),
    Scenario(
        name="Ace-king broadway: top two pair",
        board=("Ah", "Kc", "6d", "3s", "8h"),
        oop_range=("AKs", "AQs", "66", "KQs", "Qh9h", "JhTh"),
        ip_range=("AJs", "ATs", "KJs", "88", "A5s", "T9s"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets ace-king top two pair on a dry "
        "ace-king board; the caller is capped at one pair and must defend.",
    ),
    Scenario(
        name="Queen-jack broadway: two pair",
        board=("Qh", "8c", "3d", "5s", "Jc"),
        oop_range=("QJs", "AQs", "JJ", "AhTh", "Kh9h", "T9s"),
        ip_range=("QTs", "Q9s", "88", "55", "AJs", "KhTh"),
        bet_fractions=(0.5, 1.0),
        description="The river jack adds a second broadway. Hero (OOP) holds "
        "the broadway straight (T9 = eight through queen), a top set (JJ), or "
        "queen-jack two pair; the lesson is sizing with this polarized range.",
    ),
    Scenario(
        name="Ace-queen broadway: top two pair",
        board=("Ac", "Qh", "8d", "4s", "2c"),
        oop_range=("AQs", "AhKh", "QJs", "88", "KhJh", "Th9h"),
        ip_range=("AJs", "ATs", "KQs", "A5s", "Q9s", "JJ"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets ace-queen top two pair; weaker aces "
        "and queens are bluff-catchers on this dry broadway runout.",
    ),
]

# --- Low boards (no made straight). -----------------------------------------
LOW = [
    Scenario(
        name="Low board: set vs straight threat",
        board=("9d", "7s", "5c", "3h", "2d"),
        oop_range=("99", "55", "A9s", "77", "KhQh", "JhTh"),
        ip_range=("86s", "33", "T9s", "98s", "A7s", "QJs"),
        bet_fractions=(0.33, 0.75),
        description="Hero (OOP) value bets a set on a low board, but must "
        "respect IP's rivered nine-high straight (8-6 makes 5-6-7-8-9); "
        "thin value meets the nuts.",
    ),
    Scenario(
        name="Low jack-high: big pairs out of position",
        board=("Jc", "9d", "5h", "3c", "2s"),
        oop_range=("QQ", "TT", "88", "AJs"),
        ip_range=("KJs", "JTs", "99", "T8s", "A5s", "KQs"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) bets an overpair (QQ, above the board jack) "
        "and strong underpairs (TT, 88) on a low, disconnected jack-high board "
        "where thin value goes a long way.",
    ),
    Scenario(
        name="Low paired board: full house vs trips",
        board=("7h", "7d", "4c", "2s", "9h"),
        oop_range=("99", "AhAs", "KhQh", "JhTh", "Ah4h", "TT"),
        ip_range=("T7s", "97s", "44", "A4s", "QJs", "55"),
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) bets a full house (99 = nines full of sevens) "
        "or two pair (AA, TT) on a low paired board; IP hides full houses "
        "(97 = sevens full of nines, 44 = fours full of sevens) and trip "
        "sevens -- a lesson in pot control vs thin value.",
    ),
]

# --- Three-bet pots (pot = 200, steeper ranges). ----------------------------
THREE_BET = [
    Scenario(
        name="3-bet pot, dry ace-high",
        board=("Ah", "8c", "5d", "2s", "Kd"),
        oop_range=("AKs", "AQs", "KK", "QhJh", "JhTh", "QhTh"),
        ip_range=("AJs", "ATs", "QQ", "A5s", "88", "KQs"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        description="In a 3-bet pot, Hero (OOP) value bets top two pair "
        "(AK = aces and kings on an A-K board) or a set of kings (KK); "
        "IP's underpairs bluff-catch for a swollen pot.",
    ),
    Scenario(
        name="3-bet pot, low board: big hands vs sets",
        board=("Kd", "8c", "6h", "4s", "2c"),
        oop_range=("AA", "KK", "QQ", "AhQh", "JhTh", "88"),
        ip_range=("66", "44", "JJ", "A5s", "KhJh", "99"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) bets an overpair (AA), two sets (KK = kings "
        "set, 88 = eights set), and a strong underpair (QQ, below the board "
        "king) in a 3-bet pot on a low board, navigating IP's flopped sets.",
    ),
    Scenario(
        name="3-bet pot, paired ace river: trips",
        board=("Ac", "9d", "6h", "3s", "Ah"),
        oop_range=("AKs", "AQs", "99", "KhQh", "JhTh", "QhJh"),
        ip_range=("AJs", "ATs", "KK", "A5s", "QQ", "66"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        description="The river pairs the ace in a 3-bet pot. Hero (OOP) value "
        "bets trip aces with a strong kicker; 99 makes a full house (nines "
        "full of aces) for additional value. IP's big pocket pairs (KK, QQ) "
        "bluff-catch.",
    ),
    Scenario(
        name="3-bet pot, broadway: top two pair",
        board=("As", "Kd", "Qc", "7h", "2s"),
        oop_range=("AKs", "KQs", "AhJh", "QJs", "JJ", "Th9h"),
        ip_range=("AQs", "KJs", "KhTh", "A5s", "TT", "QdJd"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) value bets ace-king top two pair in a 3-bet pot "
        "on a broadway board; the caller's weaker two pair and underpairs defend.",
    ),
    Scenario(
        name="3-bet pot, two-tone: top two and strong pairs",
        board=("Ad", "Jd", "9c", "6s", "3h"),
        oop_range=("AhKh", "AJs", "99", "QcQs", "KhQh", "Th8h"),
        ip_range=("J9s", "A9s", "JJ", "A5s", "KdQd", "66"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        description="Hero (OOP) bets top two pair (AJs = aces and jacks), a "
        "set of nines (99), or strong pairs (QQ -- a bluff-catcher below the "
        "board ace) in a 3-bet pot after the flush draw bricks; IP defends "
        "with two pair (J9s, A9s) and sets (JJ = jacks set, 66 = sixes set).",
    ),
]

# All spots, concatenated. Order here is the order FELT will see them.
SPOTS = DRY + PAIRED + TWO_TONE + FLUSH + BROADWAY + LOW + THREE_BET


def spot_id(name: str) -> str:
    """Stable kebab-case id derived from the spot name, e.g.
    'Flush board: nut advantage' -> 'flush-board-nut-advantage'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def is_degenerate(ev: tuple[float, float], pot: float) -> tuple[bool, float]:
    """Decide whether a solved spot is a non-lesson.

    `ev` is each player's expected share of the pot; the two sum to `pot`.
    If the weaker side wins less than MIN_POT_SHARE_FRACTION of the pot, one
    range so dominates the other that the equilibrium collapses to "always
    fold" -- nothing to teach. Returns (degenerate?, weaker_side_fraction).
    """
    weaker_share = min(ev)
    fraction = weaker_share / pot if pot else 0.0
    return fraction < MIN_POT_SHARE_FRACTION, fraction


def precompute(iterations: int, out_path: str):
    ids = [spot_id(s.name) for s in SPOTS]
    if len(set(ids)) != len(ids):
        seen, dupes = set(), set()
        for i in ids:
            (dupes if i in seen else seen).add(i)
        raise ValueError(f"duplicate spot ids; rename these spots: {sorted(dupes)}")

    # Board-quality gate: reject awkward textures before spending time solving.
    for scenario, sid in zip(SPOTS, ids):
        bad = board_quality_violations(scenario.board)
        if bad:
            raise ValueError(
                f"{sid}: board {' '.join(scenario.board)} fails quality filter "
                f"({', '.join(bad)}); pick a cleaner board"
            )

    entries = []
    dropped = []  # (id, reason) for spots culled by the EV degeneracy guard
    total_start = time.perf_counter()
    for i, (scenario, sid) in enumerate(zip(SPOTS, ids), start=1):
        n_oop = len(expand_range(scenario.oop_range, scenario.board))
        n_ip = len(expand_range(scenario.ip_range, scenario.board))
        print(f"[{i:>2}/{len(SPOTS)}] {sid} ({n_oop}v{n_ip} combos)...", end="", flush=True)

        start = time.perf_counter()
        trainer = solve(scenario, iterations)
        ev = trainer.expected_game_value()
        elapsed = time.perf_counter() - start

        degenerate, fraction = is_degenerate(ev, scenario.pot)
        if degenerate:
            reason = (
                f"weaker side wins only {fraction:.1%} of the pot "
                f"(OOP {ev[0]:.1f} / IP {ev[1]:.1f}) -- one range dominates, "
                f"no real decision to teach"
            )
            dropped.append((sid, reason))
            print(f" {elapsed:.1f}s  DROPPED ({fraction:.1%} to weaker side)")
            continue

        print(f" {elapsed:.1f}s  (OOP pot share {ev[0]:.1f})")

        # strategy_dict provides board/pot/ranges/strategies; we add the
        # lesson metadata FELT needs on top. Schema is unchanged from before.
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
    _print_summary(out_path, entries, dropped, elapsed)


def _print_summary(out_path, entries, dropped, elapsed):
    """Print the end-of-run report: solved, dropped, and total time."""
    print(f"\nWrote {len(entries)} spots to {out_path}")
    if dropped:
        print(f"\nDropped {len(dropped)} degenerate spot(s) as non-lessons:")
        for sid, reason in dropped:
            print(f"  - {sid}: {reason}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Spots attempted : {len(entries) + len(dropped)}")
    print(f"  Spots solved    : {len(entries)}")
    print(f"  Spots dropped   : {len(dropped)} (degenerate / always-fold)")
    print(f"  Total time      : {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Precompute FELT lesson strategies")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--out", default="lesson_strategies.json")
    args = parser.parse_args()
    precompute(args.iterations, args.out)


if __name__ == "__main__":
    main()
