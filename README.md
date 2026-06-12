# River CFR Solver

A river-only Texas Hold'em solver in Python. Given a fixed 5-card board, two
player ranges, a pot size, and a set of bet sizes, it computes a Nash
equilibrium strategy using vanilla counterfactual regret minimization (CFR),
then prints a per-combo strategy table and exports the result to JSON.

It also powers [FELT](#precomputing-lesson-spots-for-felt), a React training
app: `precompute.py` solves a batch of lesson spots offline and writes them
to a single JSON file the app imports statically.

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
| `scenario.py` | The `Scenario` dataclass (board, ranges, pot, bet sizes) + range shorthand expansion |
| `solver.py` | `solve(scenario, iterations)`: validates a scenario, builds the tree, runs CFR |
| `output.py` | Strategy tables sorted by hand strength + JSON-ready dict / file export |
| `main.py` | Solves the single default scenario |
| `precompute.py` | Solves all FELT lesson spots into one `lesson_strategies.json` |

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

## Precomputing lesson spots for FELT

```bash
.venv/bin/python precompute.py --iterations 500 --out lesson_strategies.json
```

Solves every spot in `precompute.py`'s `SPOTS` list (~15 spots, ~50 seconds)
and writes one self-contained JSON file. To add a spot, append a `Scenario`
to `SPOTS` -- the id and export are derived automatically.

### `lesson_strategies.json` schema

Top level:

```jsonc
{
  "schema_version": 1,     // bump if the shape ever changes
  "iterations": 500,       // CFR iterations used per spot
  "spot_count": 15,
  "spots": [ ... ]         // one entry per lesson spot, order = SPOTS order
}
```

Each entry in `spots`:

```jsonc
{
  "id": "flush-board-nut-advantage",   // stable kebab-case key for routing
  "name": "Flush board: nut advantage",
  "description": "Three spades on board. ...",  // lesson text for the UI
  "board": ["Js", "8s", "4s", "Kh", "2c"],      // 5 cards, rank + suit (cdhs)
  "pot": 100.0,                                  // pot at start of river
  "bet_fractions": [0.5, 1.0],                   // available bet sizes (x pot)
  "raise_fractions": [1.0],                      // available raise sizes
  "iterations": 500,
  "expected_pot_share": { "OOP": 38.38, "IP": 61.62 },  // sums to pot
  "ranges": { "OOP": ["AsQs", ...], "IP": ["Ts9s", ...] },
  "strategies": {
    "OOP": {
      // keys are action histories: "root" for OOP's first decision,
      // otherwise prior actions joined with "/"
      "root": {
        "AsQs": {                       // combo: higher-ranked card first
          "strength": 6178,             // higher = stronger (max 7462)
          "hand_class": "Flush",        // display label
          "reached": true,              // see note below
          "actions": {                  // frequencies, sum to 1.0
            "check": 0.12, "bet 50": 0.07, "bet 100": 0.81
          }
        }
      },
      "check/bet 50": { ... },          // OOP facing a half-pot stab
      "bet 100/raise 100": { ... }      // OOP facing a raise after betting
    },
    "IP": {
      "check": { ... },                 // IP after OOP checks
      "bet 50": { ... },                // IP facing a half-pot bet
      "check/bet 100/raise 100": { ... }
    }
  }
}
```

Notes for the consumer:

- **Action labels** are stable strings: `check`, `fold`, `call`,
  `bet <pct>`, `raise <pct>` (pct = percent of pot, e.g. `bet 50`,
  `raise 100`). A raise label means raising BY that fraction of the
  pot-after-call.
- **`reached: false`** marks combos that never arrive at that decision point
  under equilibrium play (e.g. a hand that always folds earlier). Their
  `actions` are a meaningless uniform placeholder -- hide or grey them out.
- Every combo in a player's range appears at every one of that player's
  decision points, so lookups are total: `strategies[player][history][combo]`.

`main.py --out strategy.json` exports a single spot using the same inner
shape (everything from `board` down, without the lesson metadata).

## Assumptions & possible extensions

- Effective stacks are assumed deep enough for every line (no all-ins).
- Range weights are uniform (every combo equally likely); weighted ranges
  would only require carrying a weight into the initial reach probabilities.
- Natural next steps: CFR+ or discounted CFR for faster convergence,
  exploitability measurement (best-response computation), turn + river trees.
