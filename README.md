# River CFR Solver

A river-only Texas Hold'em solver in Python. Given a fixed 5-card board, two
player ranges, a pot size, and a set of bet sizes, it computes a Nash
equilibrium strategy using vanilla counterfactual regret minimization (CFR),
then prints a per-combo strategy table and exports the result to JSON.

Written for clarity over speed: full-tree vanilla CFR, no sampling or pruning,
with the algorithm commented step by step in `cfr.py`.

## Setup & run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --iterations 1000 --out strategy.json
.venv/bin/python -m pytest tests/
```

The default scenario (board `Ks 7h 2d 8c 3s`, a tight value-heavy OOP range vs
a wider IP range, 27 vs 25 combos) solves in ~10 seconds.

## How it's organized

| File | Purpose |
| --- | --- |
| `evaluator.py` | Wraps the `treys` hand evaluator; exposes `hand_strength` where **higher is better** |
| `game_tree.py` | Builds the river betting tree: OOP checks/bets, fold/call/raise responses, one raise cap |
| `cfr.py` | Vanilla CFR with regret matching; information sets keyed by (player, combo, action history) |
| `scenario.py` | The board, pot, bet sizes, and ranges (in shorthand like `"AA"`, `"KQs"`) |
| `output.py` | Strategy tables sorted by hand strength + JSON export |
| `main.py` | Ties it together |

## How CFR works (short version)

Both players start playing uniformly at random. Every iteration, the solver
walks the full game tree for every possible deal (every non-conflicting pair
of combos). At each decision point it tracks **regret** — how much better each
action would have done than the current mixed strategy — and the next
iteration plays actions in proportion to accumulated positive regret
("regret matching"). The *average* strategy across all iterations converges
to a Nash equilibrium in two-player zero-sum games.

Correctness is tested against a game with a known analytic solution
(`tests/test_cfr.py`): a perfectly polarized range vs a bluff catcher with a
pot-sized bet, where theory says bet all value, bluff half as often as you
value-bet, and defender calls 50% — the solver converges to exactly that.

## JSON output shape

```jsonc
{
  "board": ["Ks", "7h", "2d", "8c", "3s"],
  "pot": 100.0,
  "expected_pot_share": { "OOP": 94.42, "IP": 5.58 },
  "ranges": { "OOP": ["AcAd", ...], "IP": ["KcQc", ...] },
  "strategies": {
    "OOP": {
      "root": {
        "KcKd": {
          "strength": 5824,
          "hand_class": "Three of a Kind",
          "reached": true,
          "actions": { "check": 0.55, "bet 50": 0.0, "bet 100": 0.45 }
        }
      },
      "check/bet 50": { ... }   // keys are action histories
    },
    "IP": { ... }
  }
}
```

`reached: false` marks combos that never arrive at that decision point under
equilibrium play (their listed strategy is a meaningless uniform default).

## Assumptions & possible extensions

- Effective stacks are assumed deep enough for every line (no all-ins).
- Range weights are uniform (every combo equally likely); weighted ranges
  would only require carrying a weight into the initial reach probabilities.
- Natural next steps: CFR+ or discounted CFR for faster convergence,
  exploitability measurement (best-response computation), turn + river trees.
