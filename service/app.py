"""
Kenya's Trivia card service.

  POST /generate_card     body: {} or {"lobby_id": "abc"}   ->  {"words": [ {word, alias, hint} x 10 ]}
  GET  /health

Run (from the project root):
  CKPT=run1/best_cards.pt DATA=data/corpus uvicorn service.app:app --port 8000
Without CKPT the service still works (cards are drawn from the verified pool only), so the Django
backend can be wired up before training finishes.
"""
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
engine = Engine(
    data_dir=os.environ.get("DATA", str(ROOT / "data" / "corpus")),
    ckpt=os.environ.get("CKPT") or None,
    tokenizer=os.environ.get("TOKENIZER") or None,
    tier=os.environ.get("TIER", "serving"),
    pool_size=int(os.environ.get("POOL_SIZE", "0")),
    temperature=float(os.environ.get("GEN_TEMP", "0.7")),
)
app = FastAPI(title="Kenya's Trivia card service")


class CardRequest(BaseModel):
    lobby_id: Optional[str] = None


@app.post("/generate_card")
def generate_card(req: Optional[CardRequest] = None):
    lobby = req.lobby_id if req and req.lobby_id else None
    card, n_model = engine.build_card(lobby)
    # the response body is exactly the Card Contract; diagnostics go in a header
    return JSONResponse(card, headers={"X-Card-Source": f"model={n_model};verified_fill={10 - n_model}"})


@app.get("/health")
def health():
    return {"status": "ok", "verified_words": len(engine.verified), "model_loaded": bool(engine.model),
            "pool_ready": len(engine.pool), **engine.stats}
