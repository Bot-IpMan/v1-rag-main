import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from server.retriever import fetch_relevant

# Позволяє запускати сервіс у середовищі без доступу до інтернету.
# Якщо змінна оточення `MODEL_LOCAL_PATH` вказує на директорію з
# попередньо завантаженою моделлю, вона буде використана замість
# стандартного імені з Hugging Face Hub. Додатково можна встановити
# `HF_LOCAL_FILES_ONLY=1`, щоб примусово відключити спроби мережевих
# звернень з боку Transformers.
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
MODEL_PATH = os.getenv("MODEL_LOCAL_PATH", MODEL_NAME)
LOCAL_FILES_ONLY = os.getenv("HF_LOCAL_FILES_ONLY", "0") == "1"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, device_map="auto", local_files_only=LOCAL_FILES_ONLY
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", local_files_only=LOCAL_FILES_ONLY
)
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
