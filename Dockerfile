FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
# cron для контейнеров, запускает update_index.py в сервисе updater
ADD --checksum=sha256:cc232108374e8e2b537045f2d5eaa1229f619db314c3ba6a14d38b35dc0aa123 --chmod=755 \
    https://github.com/aptible/supercronic/releases/download/v0.2.51/supercronic-linux-amd64 \
    /usr/local/bin/supercronic

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/app/hf \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

# модель эмбеддингов скачивается при сборке, в контейнере сеть к Hugging Face не нужна
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small')"
ENV HF_HUB_OFFLINE=1

COPY build_index.py rag.py bot.py update_index.py crontab ./

RUN useradd --no-create-home app
USER app

CMD ["python", "bot.py"]
