# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.8 AS uv
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    WEIGHTFORGE_RUNS_DIR=/data/runs \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 weightforge \
    && useradd --uid 10001 --gid 10001 --create-home weightforge \
    && mkdir -p /data/runs && chown 10001:10001 /data/runs
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable && uv cache clean

USER weightforge
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["python", "-m", "uvicorn", "weightforge.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
