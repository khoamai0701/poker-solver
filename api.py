"""FastAPI backend for live CFR solve requests from the FELT trainer.

Run locally:
    uvicorn api:app --reload

The single POST /solve endpoint accepts a board, two ranges, a pot, and bet
fractions; runs the existing CFR solver; and returns the strategy dict in the
same schema as a single lesson_strategies.json spot.
"""

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from output import strategy_dict  # noqa: E402
from scenario import RANKS, SUITS, Scenario  # noqa: E402
from solver import solve as _solve  # noqa: E402

VALID_CARDS = frozenset(r + s for r in RANKS for s in SUITS)

app = FastAPI(title="Poker Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:4173",
        "https://felt-trainer.vercel.app",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    board: list[str]
    oop_range: list[str]
    ip_range: list[str]
    pot: float = 100.0
    bet_fractions: list[float] = [0.5, 1.0]
    iterations: int = 500


@app.get("/")
def health():
    return {"status": "ok", "service": "poker-solver"}


@app.post("/solve")
def solve_spot(req: SolveRequest):
    """Run CFR on a river spot and return the solved strategy."""
    # --- validate board ---
    if len(req.board) != 5:
        raise HTTPException(400, "board must have exactly 5 cards")
    if len(set(req.board)) != 5:
        raise HTTPException(400, "duplicate cards on board")
    for card in req.board:
        if card not in VALID_CARDS:
            raise HTTPException(400, f"invalid card: {card!r}")

    # --- validate ranges ---
    if not req.oop_range:
        raise HTTPException(400, "oop_range cannot be empty")
    if not req.ip_range:
        raise HTTPException(400, "ip_range cannot be empty")

    # --- validate sizing ---
    if not (req.pot > 0):
        raise HTTPException(400, "pot must be a positive number")
    if not req.bet_fractions:
        raise HTTPException(400, "bet_fractions cannot be empty")
    for f in req.bet_fractions:
        if f <= 0 or f > 10:
            raise HTTPException(400, f"bet fraction {f} is out of range (must be 0 < f ≤ 10)")

    # --- validate iterations ---
    if not (1 <= req.iterations <= 10_000):
        raise HTTPException(400, "iterations must be between 1 and 10,000")

    scenario = Scenario(
        name="live-solve",
        board=tuple(req.board),
        oop_range=tuple(req.oop_range),
        ip_range=tuple(req.ip_range),
        pot=req.pot,
        bet_fractions=tuple(req.bet_fractions),
    )

    try:
        trainer = _solve(scenario, req.iterations)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return strategy_dict(trainer, req.pot)
