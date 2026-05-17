
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

# ── 1. CREATE THE APP ────────────────────────────────────────
app = FastAPI(
    title="My LLM API",
    description="Send a question in JSON → get an AI answer in JSON",
    version="1.0.0"
)

# ── 2. LOAD THE MODEL ONCE (at startup, not per request) ─────
print("⏳ Loading TinyLlama model... (first time downloads ~2.2GB)")

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype="auto",
    device_map="auto"
)

print("✅ Model ready! Server is starting...")

# ── 3. JSON INPUT SCHEMA (what user must send) ────────────────
class QuestionRequest(BaseModel):
    question: str           # required
    topic: str = "general"  # optional, default = "general"
    max_words: int = 150    # optional, default = 150

# ── 4. JSON OUTPUT SCHEMA (what we send back) ─────────────────
class AnswerResponse(BaseModel):
    status: str
    question: str
    topic: str
    answer: str

# ── 5. HEALTH CHECK ROUTE ─────────────────────────────────────
@app.get("/")
def health_check():
    return {"status": "running", "message": "LLM API is live!"}

# ── 6. MAIN ROUTE — ASK THE LLM ──────────────────────────────
@app.post("/ask", response_model=AnswerResponse)
def ask_llm(request: QuestionRequest):

    # Error Handling — check empty question
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Error Handling — check question length
    if len(request.question) > 500:
        raise HTTPException(
            status_code=400,
            detail="Question too long. Keep it under 500 characters."
        )

    # f-string — build the prompt dynamically
    system_prompt = f"You are a helpful AI expert on the topic of {request.topic}. Answer clearly and simply."
    user_prompt   = f"Answer in simple words (max {request.max_words} words): {request.question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ]

    # Call the LLM — wrapped in try/except for safety
    try:
        output = pipe(
            messages,
            max_new_tokens=request.max_words * 2,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        answer = output[0]["generated_text"][-1]["content"]

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM call failed: {str(e)}"
        )

    # Return JSON response
    return AnswerResponse(
        status="success",
        question=request.question,
        topic=request.topic,
        answer=answer.strip()
    )

# ── 7. RUN THE SERVER ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("llm_api:app", host="0.0.0.0", port=8000, reload=False)