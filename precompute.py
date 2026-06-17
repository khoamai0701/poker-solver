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

from evaluator import semantic_strength
from output import strategy_dict
from scenario import RANKS, Scenario, expand_range
from solver import solve

SCHEMA_VERSION = 1

# Every label evaluator.semantic_strength can emit. featured_hand_strength on a
# spot must draw from this set (plus the "bluff" alias below).
VALID_SEMANTIC_STRENGTHS = {
    "high_card", "underpair", "weak_pair", "second_pair", "top_pair",
    "overpair", "two_pair", "trips", "set", "straight", "flush",
    "full_house", "quads", "straight_flush",
}

# Human-friendly aliases accepted in featured_hand_strength, mapped to the
# canonical semantic_strength label. Kept in sync with FELT's solverSpots.js.
FEATURED_ALIASES = {"bluff": "high_card"}


def _resolve_featured(label: str) -> str:
    return FEATURED_ALIASES.get(label, label)


# Textbook "Read More" blurbs keyed by concept name. Each spot references one
# of these so the text stays consistent and is written in one place.
_READ_MORE = {
    "Range Advantage": (
        "Range advantage means your hand distribution is stronger than your opponent's on a "
        "specific board, giving you more strong hands relative to your overall range. "
        "Solvers exploit this by betting frequently and at larger sizes, since the opponent "
        "must defend a wide value range, not just the nuts. "
        "Failing to press range advantage — by checking too often or sizing too small — "
        "leaves significant EV on the table."
    ),
    "Kicker Battle": (
        "A kicker battle occurs when both players hold the same hand category and the winner "
        "is decided solely by the unpaired hole card. "
        "Solvers bet thin in these spots because the opponent's calling range holds the same "
        "category but weaker kickers, but sizing must stay small to avoid inflating the pot "
        "against the rare dominating hand. "
        "These are among the most common thin-value situations on ace-high or paired boards."
    ),
    "Thin Value": (
        "Thin value betting means betting a hand that is only a small favorite against the "
        "opponent's calling range. "
        "The bet extracts chips from slightly worse hands that still call, while accepting "
        "the occasional loss to a better hand in villain's range. "
        "Sizing down is critical: a large bet for thin value folds out the weaker holdings "
        "you need to pay you off, turning a profitable spot into a break-even or losing one."
    ),
    "Polarization": (
        "A polarized betting range contains strong value hands and bluffs with few "
        "medium-strength holdings in between. "
        "Betting a polarized range forces the opponent into a binary decision — value or "
        "bluff — rather than letting them call comfortably with medium-strength hands. "
        "The bluff frequency in a balanced polarized range is set so the opponent is "
        "indifferent to calling: the bluff-to-value ratio mirrors the pot odds offered."
    ),
    "Nut Advantage": (
        "Nut advantage means holding a disproportionate share of the strongest possible "
        "hands on a given board. "
        "A player with nut advantage can bet large and frequently because even when called "
        "by strong hands, the opponent cannot raise without maximum exposure to the nuts. "
        "This advantage is most pronounced on flush and paired boards, where one player's "
        "range connects far more strongly with the runout."
    ),
    "Bluff Catching": (
        "Bluff catching means calling down with a hand that loses to all value bets but "
        "beats all bluffs. "
        "The decision is driven by pot odds and the estimated frequency villain bluffs: if "
        "villain bluffs more often than the break-even frequency implied by the pot odds, "
        "calling is profitable regardless of hand strength. "
        "Bluff catchers should rarely raise, because raising prices out the bluffs you're "
        "trying to capture and only gets called by better."
    ),
}

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
# The curated lesson spots, grouped by texture / concept.
# ---------------------------------------------------------------------------

# --- Dry, disconnected boards (single-raised pots). -------------------------
DRY = [
    Scenario(
        name="King-high dry: range advantage",
        featured_hand_strength=("overpair", "set", "top_pair"),
        board=("Ks", "7h", "2d", "8c", "3s"),
        oop_range=("AA", "KK", "AK", "88", "77"),
        ip_range=("KQs", "KJs", "99", "87s", "A7s", "QJs", "T9s"),
        bet_fractions=(0.5, 1.0),
        concept="Range Advantage",
        read_more=_READ_MORE["Range Advantage"],
        description=(
            "Your range dominates villain's bluff-catching range on this dry king-high board — "
            "sets, overpairs, and top pairs all sit ahead of villain's capped holdings. "
            "Bet large and often; range advantage this clean means you can pressure "
            "with your entire value range."
        ),
    ),
    Scenario(
        name="Ace-high dry: top-pair kicker battle",
        featured_hand_strength=("top_pair",),
        board=("Ah", "Kd", "7c", "4s", "2h"),
        oop_range=("AQs", "AJs", "ATs", "KQs", "77", "44"),
        ip_range=("KK", "22", "A9s", "A8s", "QQ", "JJ", "KJs"),
        bet_fractions=(0.5, 1.0),
        concept="Kicker Battle",
        read_more=_READ_MORE["Kicker Battle"],
        description=(
            "Both you and villain hold aces here, so the kicker determines who wins. "
            "Bet small for thin value — you're ahead of weaker aces, but keep the size down "
            "to avoid building a big pot against a better kicker."
        ),
    ),
    Scenario(
        name="Queen-high dry: two-pair value",
        featured_hand_strength=("two_pair", "set"),
        board=("Qd", "7s", "3h", "Jc", "2d"),
        oop_range=("QJs", "AQs", "JJ", "33", "Kh9h", "T9s"),
        ip_range=("QTs", "Q9s", "77", "AJs", "KJs", "A5s"),
        bet_fractions=(0.5, 1.0),
        concept="Polarization",
        read_more=_READ_MORE["Polarization"],
        description=(
            "Your two pair or set is the clear value hand on this disconnected board, "
            "and missed straight draws become natural bluffs with no showdown value. "
            "Polarize your betting range — bet large with your value and bluffs "
            "and let villain's top pair make a tough call."
        ),
    ),
    Scenario(
        name="King-queen broadway dry",
        featured_hand_strength=("two_pair", "set", "bluff"),
        board=("Kd", "Qc", "5h", "2s", "7d"),
        oop_range=("KQs", "AKs", "55", "AhJh", "Th9h", "QJs"),
        ip_range=("KJs", "KTs", "QJs", "77", "A5s", "JhTh"),
        bet_fractions=(0.5, 1.0),
        concept="Polarization",
        read_more=_READ_MORE["Polarization"],
        description=(
            "Your range holds a nut advantage on this dry broadway board — "
            "top two pair and sets sit ahead of villain's weaker kings and second pair. "
            "Missed draws in your range have no showdown value, "
            "making them the natural bluffs to balance your value bets."
        ),
    ),
    Scenario(
        name="Jack-high dry: thin value",
        featured_hand_strength=("top_pair",),
        board=("Jd", "6c", "2h", "9s", "4d"),
        oop_range=("JTs", "AJs", "99", "44", "KhQh", "Ah5h"),
        ip_range=("J9s", "QJs", "66", "88", "T9s", "A6s"),
        bet_fractions=(0.5,),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "Top pair is a thin value hand on this low board, but villain's weaker jacks "
            "and underpairs will call a small bet and fold to a large one. "
            "Size down to extract from the hands that pay you off."
        ),
    ),
    Scenario(
        name="Ten-high dry: top set value",
        featured_hand_strength=("set",),
        board=("Td", "5c", "2h", "7s", "3d"),
        oop_range=("TT", "55", "ATs", "KhQh", "Ah9h", "QJs"),
        ip_range=("T9s", "T8s", "77", "33", "A5s", "JhQh"),
        bet_fractions=(0.5, 1.0),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "Your set is the dominant value hand on this ten-high board — "
            "well ahead of villain's weaker tens and pairs. "
            "Bet large to maximize extraction; the overcard hands that missed "
            "are the natural bluffs to balance your sizing."
        ),
    ),
    Scenario(
        name="Ace-high dry: weak-ace defense",
        featured_hand_strength=("top_pair",),
        board=("As", "8d", "3c", "9h", "2s"),
        oop_range=("A7s", "A5s", "99", "33", "KQs", "QJs"),
        ip_range=("ATs", "A9s", "88", "KhJh", "QhTh", "77"),
        bet_fractions=(0.5,),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "You hold a weak top pair on an ace-high board where villain's range "
            "includes stronger aces. "
            "Bet small to extract from the hands you beat and deny equity, "
            "without bloating the pot against a better ace."
        ),
    ),
]

# --- Paired boards. ---------------------------------------------------------
PAIRED = [
    Scenario(
        name="Paired aces: kicker war",
        featured_hand_strength=("trips", "full_house", "bluff"),
        board=("Ah", "Ad", "7c", "4s", "2h"),
        oop_range=("AKs", "AQs", "77", "KhQh", "JhTh", "KhJh"),
        ip_range=("AJs", "ATs", "22", "99", "A5s", "KQs"),
        bet_fractions=(0.5, 1.0),
        concept="Kicker Battle",
        read_more=_READ_MORE["Kicker Battle"],
        description=(
            "On a paired-aces board, everyone has trip aces — "
            "the pot goes to the better kicker or full house. "
            "Value bet with confidence when yours is at the top; "
            "villain's weaker trips are stuck paying you off or folding."
        ),
    ),
    Scenario(
        name="Paired sixes: full house and big pairs",
        featured_hand_strength=("full_house", "quads", "two_pair"),
        board=("6s", "6d", "Kc", "3h", "2s"),
        oop_range=("KK", "AA", "66", "QhJh", "AhTh", "QQ"),
        ip_range=("K9s", "KJs", "99", "A6s", "55", "JTs"),
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "Your full house completely dominates villain's one-pair range "
            "on this paired-six board. "
            "Bet large; villain's kings and pairs have to call "
            "without any ability to fight back."
        ),
    ),
    Scenario(
        name="Paired kings: trips vs underpair",
        featured_hand_strength=("trips", "full_house"),
        board=("Ks", "Kh", "9d", "4c", "7s"),
        oop_range=("KQs", "99", "AhQh", "JhTh", "QhJh", "AhJh"),
        ip_range=("KJs", "KTs", "QQ", "A9s", "JJ", "TT"),
        bet_fractions=(0.33, 1.0),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "Your trip kings are ahead of every pocket pair villain holds — "
            "they're all underpairs to the board pair. "
            "Choose a small size for thin value from underpairs "
            "or a large size to apply pressure."
        ),
    ),
    Scenario(
        name="Board pairs eights: boats freeze pairs",
        featured_hand_strength=("full_house", "quads", "two_pair"),
        board=("As", "8c", "5d", "8h", "2s"),
        oop_range=("AKs", "AQs", "55", "88", "AA"),
        ip_range=("98s", "T8s", "A8s", "TT", "99", "A7s", "KQs"),
        bet_fractions=(0.5, 1.0),
        concept="Polarization",
        read_more=_READ_MORE["Polarization"],
        description=(
            "The river pairing the eight reshuffled the hand rankings — "
            "full houses and quads jumped to the top, making formerly strong hands "
            "like top two pair into bluff-catchers. "
            "Recognize the shift: what was a value hand before the river may now be a call."
        ),
    ),
    Scenario(
        name="Trips on board: kicker plays",
        featured_hand_strength=("full_house", "trips"),
        board=("4c", "4d", "4h", "Qs", "9d"),
        oop_range=("AhKh", "AQs", "QQ", "KhJh", "Th8h", "99"),
        ip_range=("KQs", "Q9s", "JJ", "A5s", "TT", "JhTh"),
        bet_fractions=(0.33, 1.0),
        concept="Kicker Battle",
        read_more=_READ_MORE["Kicker Battle"],
        description=(
            "When the board shows trips, everyone holds at least trip fours — "
            "the pot goes to the best kicker or full house. "
            "Value bet when your full house or kicker beats the calling range; "
            "check when villain's kickers have you at a disadvantage."
        ),
    ),
    Scenario(
        name="3-bet pot, paired queens: full house vs trips",
        featured_hand_strength=("full_house", "two_pair", "trips"),
        board=("Qd", "Qs", "6h", "3c", "Jd"),
        oop_range=("AA", "KK", "AhQh", "JJ", "KhTh", "AdKd"),
        ip_range=("QJs", "Q9s", "TT", "A5s", "KQs", "99"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "In this bloated 3-bet pot, your jacks full is a premium hand, "
            "but villain's slow-played trip queens lurk in the calling range. "
            "Bet for value, but respect large raises — villain's trips show up "
            "in the range more often than it feels."
        ),
    ),
]

# --- Two-tone boards (max two of a suit -> no flush possible). ---------------
TWO_TONE = [
    Scenario(
        name="Two-tone broadway brick: two-pair value",
        featured_hand_strength=("two_pair", "set", "bluff"),
        board=("Ad", "Qd", "8c", "4s", "3h"),
        oop_range=("AQs", "AcKc", "QJs", "88", "KdJd", "Th9h"),
        ip_range=("AJs", "ATs", "QdTd", "44", "Jd9d", "KhJh"),
        bet_fractions=(0.75,),
        concept="Polarization",
        read_more=_READ_MORE["Polarization"],
        description=(
            "The flush draw bricked, leaving your top two pair or set as the clear value hand "
            "and your missed draws as the natural bluffs. "
            "Villain can't have a flush and is deciding whether one pair "
            "is worth calling your bet."
        ),
    ),
    Scenario(
        name="Two-tone king board: top pair",
        featured_hand_strength=("top_pair", "set"),
        board=("Kd", "Td", "7c", "4s", "2h"),
        oop_range=("AKs", "KQs", "TT", "KhQh", "JhTh", "99"),
        ip_range=("KJs", "KTs", "QJs", "A5s", "QdJd", "88"),
        bet_fractions=(0.5, 1.0),
        concept="Range Advantage",
        read_more=_READ_MORE["Range Advantage"],
        description=(
            "The flush draw missed and your top pair or set is the dominant hand — "
            "villain's range is capped at second pair with no draw to hide behind. "
            "Value bet comfortably; villain is stuck calling with second pair or worse."
        ),
    ),
    Scenario(
        name="Two-tone queen-high: set and top pair",
        featured_hand_strength=("set", "top_pair"),
        board=("Qh", "9h", "5c", "3d", "2s"),
        oop_range=("AQs", "QJs", "99", "55", "KhJh", "Th8h"),
        ip_range=("QTs", "Q9s", "JJ", "A5s", "T9s", "K9s"),
        bet_fractions=(0.5,),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "With no flush possible on this board, your set or top pair is the best hand "
            "and villain holds weaker queens and underpairs. "
            "A small size extracts thin value — villain calls what they can beat "
            "but folds to a large overbet."
        ),
    ),
    Scenario(
        name="Two-tone ace-ten: two pair",
        featured_hand_strength=("two_pair", "set"),
        board=("As", "Ts", "6d", "4c", "2h"),
        oop_range=("AQs", "AJs", "ATs", "66", "KhQh", "Jh9h"),
        ip_range=("A6s", "A4s", "TT", "KsQs", "QhJh", "T9s"),
        bet_fractions=(0.33, 0.75),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "Your two pair leads a dry board where villain holds mostly weaker aces "
            "and the occasional set — no flush or straight to fear. "
            "Pick your size based on what you want: a smaller bet gets called by weak aces, "
            "a larger bet extracts more from sets and gives up the marginal calls."
        ),
    ),
    Scenario(
        name="Two-tone jack-ten: two pair",
        featured_hand_strength=("two_pair", "set"),
        board=("Jc", "Tc", "6h", "3s", "2d"),
        oop_range=("JTs", "AJs", "ATs", "KdQd", "Qh9h", "33"),
        ip_range=("J9s", "T9s", "QJs", "A5s", "KsJs", "88"),
        bet_fractions=(0.5,),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "The straight draw missed and your two pair is the top value hand "
            "on this connected board. "
            "Villain's weaker jacks and second pair have to call and pay you off — "
            "bet for value."
        ),
    ),
]

# --- Monotone / flush-possible boards (exactly three of a suit). ------------
FLUSH = [
    Scenario(
        name="Flush possible: flush vs made hands",
        featured_hand_strength=("flush",),
        board=("Qh", "9h", "5h", "Ad", "2c"),
        oop_range=("KhJh", "Th8h", "AcQc", "AsQs", "AA", "KQs"),
        ip_range=("7h6h", "AhJc", "QJs", "99", "55", "JTs"),
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "When the flush completes, it becomes the nuts and every non-flush hand — "
            "two pair, sets, overpairs — becomes a bluff-catcher. "
            "Your flush bets for value; villain's strong non-flush hands "
            "can only hope you're bluffing."
        ),
    ),
    Scenario(
        name="Flush board: nut advantage",
        featured_hand_strength=("flush",),
        board=("Js", "8s", "4s", "Kh", "2c"),
        oop_range=("AsQs", "QsTs", "9s7s", "KK", "AJs", "99"),
        ip_range=("Ts9s", "7s6s", "6s5s", "KQs", "JTs", "88"),
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "On a flush board, the nut flush owns every smaller flush — "
            "villain's medium flushes are trapped and have to pay you off. "
            "Bet large; villain's smaller flushes can't fold "
            "and the bigger the size, the more you extract."
        ),
    ),
    Scenario(
        name="King-high flush board",
        featured_hand_strength=("flush",),
        board=("Kh", "Th", "6h", "4s", "2c"),
        oop_range=("AhQh", "Jh9h", "KK", "TT", "AcKc", "QsJs"),
        ip_range=("Qh8h", "7h5h", "KQs", "66", "A5s", "T9s"),
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "Even a top set is just a bluff-catcher once the flush comes in — "
            "the made flush wins at showdown regardless of what else the board shows. "
            "Your flush bets for value while villain's sets and strong pairs "
            "are forced to call and hope you're bluffing."
        ),
    ),
    Scenario(
        name="Low flush board: nut flush vs set",
        featured_hand_strength=("flush", "set"),
        board=("9c", "6c", "3c", "Ah", "Kd"),
        oop_range=("AcQc", "Tc8c", "99", "AsAd", "KhQh", "JhTh"),
        ip_range=("Kc7c", "6c5c", "66", "A6s", "QhJh", "T9s"),
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "On a flush board, the nut flush and a set play completely different roles — "
            "the flush bets for value, the set can only bluff-catch. "
            "Know which role your hand plays and act accordingly."
        ),
    ),
    Scenario(
        name="Broadway flush board",
        featured_hand_strength=("flush", "two_pair", "set"),
        board=("Kd", "Qd", "7d", "4s", "2c"),
        oop_range=("AdJd", "Td9d", "KQs", "QQ", "AcKc", "JhTh"),
        ip_range=("Jd8d", "9d8d", "KJs", "77", "A5s", "T9s"),
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "On a high flush board, your flush is the primary value hand — "
            "two pair and sets lose to any flush you hold. "
            "Your key decision is your flush's relative strength: "
            "the nut flush bets large, smaller flushes check more often."
        ),
    ),
]

# --- Broadway boards (high cards, no made straight, flush not live). --------
BROADWAY = [
    Scenario(
        name="KQJ broadway: straights and two pair",
        featured_hand_strength=("straight", "two_pair"),
        board=("Kh", "Qc", "Jd", "5s", "2h"),
        oop_range=("AKs", "KQs", "AhTh", "QhJh", "AA", "JJ"),
        ip_range=("AQs", "KJs", "ATs", "A5s", "TT", "QJs"),
        bet_fractions=(0.5, 1.0),
        concept="Polarization",
        read_more=_READ_MORE["Polarization"],
        description=(
            "On a K-Q-J board, the ace-ten broadway straight is the nuts and every "
            "strong hand below it — sets, overpairs, two pair — is a bluff-catcher against it. "
            "When you hold the straight, bet large; when villain raises on this board, "
            "the straight is heavily represented in their range."
        ),
    ),
    Scenario(
        name="Ace-king broadway: top two pair",
        featured_hand_strength=("two_pair", "set"),
        board=("Ah", "Kc", "6d", "3s", "8h"),
        oop_range=("AKs", "AQs", "66", "KQs", "Qh9h", "JhTh"),
        ip_range=("AJs", "ATs", "KJs", "88", "A5s", "T9s"),
        bet_fractions=(0.5, 1.0),
        concept="Range Advantage",
        read_more=_READ_MORE["Range Advantage"],
        description=(
            "Your top two pair or set dominates this board and villain is capped at single pair — "
            "no flush or straight threatens you. "
            "Bet for value; villain's one-pair hands are bluff-catching, not trapping."
        ),
    ),
    Scenario(
        name="Queen-jack broadway: straights, sets & two pair",
        featured_hand_strength=("straight", "set", "two_pair"),
        board=("Qh", "8c", "3d", "5s", "Jc"),
        oop_range=("QJs", "AQs", "JJ", "AhTh", "Kh9h", "T9s"),
        ip_range=("QTs", "Q9s", "88", "55", "AJs", "KhTh"),
        bet_fractions=(0.5, 1.0),
        concept="Polarization",
        read_more=_READ_MORE["Polarization"],
        description=(
            "Your range is polarized with straights, sets, and two pair, "
            "while villain holds weaker queens and has to call without knowing "
            "which value hand you hold. "
            "Bet large — any of your strong hands are ahead of most of what villain can call with."
        ),
    ),
    Scenario(
        name="Ace-queen broadway: top two pair",
        featured_hand_strength=("two_pair", "set"),
        board=("Ac", "Qh", "8d", "4s", "2c"),
        oop_range=("AQs", "AhKh", "QJs", "88", "KhJh", "Th9h"),
        ip_range=("AJs", "ATs", "KQs", "A5s", "Q9s", "JJ"),
        bet_fractions=(0.5, 1.0),
        concept="Range Advantage",
        read_more=_READ_MORE["Range Advantage"],
        description=(
            "Your top two pair or set leads a board where villain is capped at single-pair hands — "
            "no hidden straights or flushes lurk. "
            "Value bet with confidence; villain's aces and queens are paying you off or folding."
        ),
    ),
]

# --- Low boards (no made straight). -----------------------------------------
LOW = [
    Scenario(
        name="Low board: set vs straight threat",
        featured_hand_strength=("set",),
        board=("9d", "7s", "5c", "3h", "2d"),
        oop_range=("99", "55", "A9s", "77", "KhQh", "JhTh"),
        ip_range=("86s", "33", "T9s", "98s", "A7s", "QJs"),
        bet_fractions=(0.33, 0.75),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "Your set is strong on this low board, but the eight-six straight "
            "lurks in villain's range and beats you cleanly. "
            "Size carefully — a smaller bet gets value from worse pairs "
            "without building a pot you'd lose to the straight."
        ),
    ),
    Scenario(
        name="Low jack-high: big pairs out of position",
        featured_hand_strength=("overpair", "underpair", "top_pair"),
        board=("Jc", "9d", "5h", "3c", "2s"),
        oop_range=("QQ", "TT", "88", "AJs"),
        ip_range=("KJs", "JTs", "99", "T8s", "A5s", "KQs"),
        bet_fractions=(0.5, 1.0),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "On this low, disconnected board your strong pairs are ahead of nearly "
            "everything villain holds — the board misses most of their range. "
            "Even underpairs extract thin value here; bet for what the board allows."
        ),
    ),
    Scenario(
        name="Low paired board: full house vs trips",
        featured_hand_strength=("full_house", "two_pair"),
        board=("7h", "7d", "4c", "2s", "9h"),
        oop_range=("99", "AhAs", "KhQh", "JhTh", "Ah4h", "TT"),
        ip_range=("T7s", "97s", "44", "A4s", "QJs", "55"),
        bet_fractions=(0.5, 1.0),
        concept="Thin Value",
        read_more=_READ_MORE["Thin Value"],
        description=(
            "Your full house looks dominant, but villain's range hides its own "
            "full houses and trips that beat you. "
            "Bet for value, but treat a raise as a warning — villain's calling range "
            "on a paired low board is much stronger than it appears."
        ),
    ),
]

# --- Three-bet pots (pot = 200, steeper ranges). ----------------------------
THREE_BET = [
    Scenario(
        name="3-bet pot, dry ace-high",
        featured_hand_strength=("two_pair", "set", "bluff"),
        board=("Ah", "8c", "5d", "2s", "Kd"),
        oop_range=("AKs", "AQs", "KK", "QhJh", "JhTh", "QhTh"),
        ip_range=("AJs", "ATs", "QQ", "A5s", "88", "KQs"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        concept="Range Advantage",
        read_more=_READ_MORE["Range Advantage"],
        description=(
            "In a 3-bet pot, your nut advantage is at its most powerful — "
            "inflated pot sizes mean each bet applies maximum pressure to "
            "villain's underpairs and weaker aces. "
            "Size up aggressively; the big pot rewards value betting and makes "
            "villain's marginal hands extremely difficult to call."
        ),
    ),
    Scenario(
        name="3-bet pot, low board: big hands vs sets",
        featured_hand_strength=("overpair", "set", "underpair"),
        board=("Kd", "8c", "6h", "4s", "2c"),
        oop_range=("AA", "KK", "QQ", "AhQh", "JhTh", "88"),
        ip_range=("66", "44", "JJ", "A5s", "KhJh", "99"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        concept="Bluff Catching",
        read_more=_READ_MORE["Bluff Catching"],
        description=(
            "In a 3-bet pot on a low board, villain's range frequently contains "
            "flopped sets that crush your overpairs — the big pot makes those mistakes very costly. "
            "Value bet the nutted hands confidently, but respect a raise; "
            "villain's set is live on this texture."
        ),
    ),
    Scenario(
        name="3-bet pot, paired ace river: trips",
        featured_hand_strength=("trips", "full_house", "bluff"),
        board=("Ac", "9d", "6h", "3s", "Ah"),
        oop_range=("AKs", "AQs", "99", "KhQh", "JhTh", "QhJh"),
        ip_range=("AJs", "ATs", "KK", "A5s", "QQ", "66"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        concept="Nut Advantage",
        read_more=_READ_MORE["Nut Advantage"],
        description=(
            "The river pairing the ace gives your trips a dominant position in this bloated pot — "
            "villain's overpairs are stuck bluff-catching with no way to raise profitably. "
            "Extract maximum value at a large size; villain's options narrow to call or fold."
        ),
    ),
    Scenario(
        name="3-bet pot, broadway: top two pair",
        featured_hand_strength=("two_pair",),
        board=("As", "Kd", "Qc", "7h", "2s"),
        oop_range=("AKs", "KQs", "AhJh", "QJs", "JJ", "Th9h"),
        ip_range=("AQs", "KJs", "KhTh", "A5s", "TT", "QdJd"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        concept="Range Advantage",
        read_more=_READ_MORE["Range Advantage"],
        description=(
            "In a 3-bet pot on a broadway board, your top two pair is the best made hand "
            "and villain is capped at weaker two pair or one pair. "
            "Press the advantage — the large pot rewards value betting, "
            "and villain can't reasonably fold to a standard bet."
        ),
    ),
    Scenario(
        name="3-bet pot, two-tone: top two and strong pairs",
        featured_hand_strength=("two_pair", "set", "underpair"),
        board=("Ad", "Jd", "9c", "6s", "3h"),
        oop_range=("AhKh", "AJs", "99", "QcQs", "KhQh", "Th8h"),
        ip_range=("J9s", "A9s", "JJ", "A5s", "KdQd", "66"),
        pot=200.0,
        bet_fractions=(0.5, 1.0),
        concept="Range Advantage",
        read_more=_READ_MORE["Range Advantage"],
        description=(
            "After the draw bricks, both you and villain hold strong made hands in this "
            "bloated pot — but your top two pair and sets sit at the top of both ranges. "
            "In a spot where both sides are strong, the player with the slightly better "
            "range at the top wins; that's you."
        ),
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

    # Featured-hand gate: every spot must name the concept hands it teaches, and
    # each named strength must actually exist in the OOP range. This catches typos
    # and stale featured lists at build time rather than as a silent FELT fallback.
    for scenario, sid in zip(SPOTS, ids):
        featured = scenario.featured_hand_strength
        if not featured:
            raise ValueError(f"{sid}: featured_hand_strength is empty; name the lesson's concept hands")
        wanted = set()
        for label in featured:
            resolved = _resolve_featured(label)
            if resolved not in VALID_SEMANTIC_STRENGTHS:
                raise ValueError(
                    f"{sid}: unknown featured_hand_strength {label!r} "
                    f"(valid: {sorted(VALID_SEMANTIC_STRENGTHS)} or aliases {sorted(FEATURED_ALIASES)})"
                )
            wanted.add(resolved)
        available = {
            semantic_strength(list(c), scenario.board)
            for c in expand_range(scenario.oop_range, scenario.board)
        }
        missing = wanted - available
        if missing:
            raise ValueError(
                f"{sid}: featured strengths {sorted(missing)} never occur in the OOP "
                f"range on this board (range produces {sorted(available)}); fix the "
                f"featured list or the range"
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

        entry = {
            "id": sid,
            "name": scenario.name,
            "description": scenario.description,
            "concept": scenario.concept,
            "read_more": scenario.read_more,
            "featured_hand_strength": list(scenario.featured_hand_strength),
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
