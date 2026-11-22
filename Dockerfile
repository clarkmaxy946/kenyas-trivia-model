# Build from the project root:  docker build -t kenyas-trivia .
# Run:  docker run -p 8000:8000 -e CKPT=/app/model/best_cards.pt kenyas-trivia
FROM python:3.11-slim
WORKDIR /app
COPY service/requirements.txt /tmp/req.txt
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/req.txt
COPY service/ /app/service/
COPY train/cards.py train/model.py train/generate.py train/tokenizer.json /app/train/
COPY data/corpus/entries_*.jsonl /app/data/corpus/
COPY model/ /app/model/
ENV DATA=/app/data/corpus TOKENIZER=/app/train/tokenizer.json POOL_SIZE=8
EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
