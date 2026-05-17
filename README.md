# My LLM API

A REST API that lets you send a question in JSON and get an AI-generated answer in JSON — powered by **TinyLlama** running locally via **FastAPI**.

---

## What It Does

- Loads **TinyLlama 1.1B Chat** once at startup (no reload per request)
- Accepts `POST /ask` with a question, optional topic, and max word limit
- Returns structured JSON: status, question, topic, and answer
- Includes validation (empty question, length limit) and error handling
- Health check at `GET /` to confirm the server is running

**Flow:** Client sends JSON → FastAPI validates → TinyLlama generates answer → JSON response

---

## Tech Stack

| Technology | Role |
|------------|------|
| **Python 3.10+** | Runtime |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **Pydantic** | Request/response JSON schemas |
| **Hugging Face Transformers** | Load and run TinyLlama |
| **TinyLlama 1.1B Chat** | Local LLM (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) |
| **PyTorch** | Model inference |
| **Accelerate** | Device / dtype auto-selection |

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/ask` | Send a question, get an AI answer |

### Request body (`POST /ask`)

```json
{
  "question": "What is machine learning?",
  "topic": "AI",
  "max_words": 150
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `question` | Yes | — | Your question (max 500 characters) |
| `topic` | No | `"general"` | Topic for the system prompt |
| `max_words` | No | `150` | Approximate answer length hint |

### Sample response

```json
{
  "status": "success",
  "question": "What is machine learning?",
  "topic": "AI",
  "answer": "Machine learning is a branch of AI where computers learn patterns from data instead of being explicitly programmed for every task."
}
```

---

## How to Run

### 1. Install dependencies

```bash
pip install fastapi uvicorn transformers torch accelerate pydantic
```

CPU-only (optional):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Start the server

```bash
python llm_api.py
```

First run downloads the model (~2.2 GB). Wait for:

```
✅ Model ready! Server is starting...
```

### 3. Test in browser

Open: [http://localhost:8000/docs](http://localhost:8000/docs) — interactive Swagger UI

### 4. Test with curl

```bash
curl -X POST http://localhost:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is Python?\", \"topic\": \"programming\"}"
```

---

## Project Structure

```
basiclearning/
├── llm_api.py    # FastAPI app + TinyLlama pipeline
└── README.md     # This file
```

---

## Author

**Tasmia Bashir** — AIML learning project

---

## License

Educational / portfolio use.
