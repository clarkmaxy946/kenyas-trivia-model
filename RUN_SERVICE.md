# Running the card service on your laptop

One-time (Command Prompt, project root):
    pip install fastapi uvicorn torch tokenizers numpy

## 1. Without a model (works today; cards come from the verified pool)
    set DATA=data\corpus
    python -m uvicorn service.app:app --port 8000
Open http://127.0.0.1:8000/docs, click POST /generate_card -> Try it out -> Execute.
Point your Django backend at http://127.0.0.1:8000/generate_card

## 2. With the trained model (after Kaggle training)
Put best_cards.pt in the folder `model\`, then:
    set DATA=data\corpus
    set CKPT=model\best_cards.pt
    set POOL_SIZE=8
    python -m uvicorn service.app:app --port 8000
POOL_SIZE=8 keeps 8 model-proposed cards ready in the background so requests answer fast.

## Request / response (the Card Contract)
    POST /generate_card        {}    or    {"lobby_id": "lobby-42"}
    -> {"words": [ {"word": "...", "alias": "...", "hint": "..."}  x 10 ]}
With a lobby_id a word is never repeated until the lobby has seen every word in the pool.
Header X-Card-Source shows how many words came from the model vs the verified fill.

## Evaluation
    python eval\loadtest.py --users 1 5 10 --requests 10        (latency table)
    python eval\rating_sheet.py --cards 10 --out eval\ratings.csv   (human ratings)
    python eval\rating_sheet.py --summarise eval\ratings.csv

## Docker (later deployment)
    docker build -t kenyas-trivia .
    docker run -p 8000:8000 -e CKPT=/app/model/best_cards.pt kenyas-trivia
