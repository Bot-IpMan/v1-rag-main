# v1-rag-main

Готовий мінімальний стек Retrieval-Augmented Generation (RAG) на базі Qwen 2.5‑7B‑Instruct і Qdrant.

## Швидкий старт

```bash
# 1. Запустити сервіси
docker compose up -d --build

# 2. Проіндексувати власні документи
docker compose run --rm rag-api python ingest.py --source_dir data/knowledge

# 3. Запит до API
curl -X POST http://localhost:8000/v1/chat/completions              -H "Content-Type: application/json"              -d '{"question":"Як вирощувати горох гідропонно?"}'
```

### Робота без доступу до інтернету

1. Завантажте модель Qwen заздалегідь і розпакуйте її в директорію, наприклад, `models/qwen`.
2. Перед запуском сервісу встановіть змінні оточення:

```bash
export MODEL_LOCAL_PATH=models/qwen
export HF_LOCAL_FILES_ONLY=1
```

Тепер `rag-api` використовуватиме локальні файли моделі й не намагатиметься звертатися до мережі.

## Структура

- `data/knowledge` — сюди кладемо `.md` / `.txt` файли з контентом.
- `ingest.py` — перетворює документи у вектори й кладе в Qdrant.
- `server/retriever.py` — пошук релевантних фрагментів.
- `server/api_server.py` — FastAPI + LLM, який отримує контекст і генерує відповідь.
```
