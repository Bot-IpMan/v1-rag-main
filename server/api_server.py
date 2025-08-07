import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from retriever import fetch_relevant

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    top_k: int | None = 4

def build_prompt(question: str, context: list[str]) -> str:
    ctx_block = "\n\n".join(context)
    system = (
        "Відповідай українською, використовуючи наданий контекст. "
        "Якщо відповіді немає у контексті — скажи чесно, що не знаєш."
    )
    return f"<|system|>\n{system}\n<|context|>\n{ctx_block}\n<|user|>\n{question}\n<|assistant|>"

@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    ctx = fetch_relevant(req.question, req.top_k)
    prompt = build_prompt(req.question, ctx)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512, streamer=streamer)
    answer = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return {"answer": answer, "context_docs": ctx}
