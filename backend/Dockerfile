# Build context: repository root  (docker build -f backend/Dockerfile .)
FROM docker.arvancloud.ir/python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The docker CLI starts MCP servers inside each sandbox (`docker exec -i`). Copy it from the official image:
# multi-arch (amd64/arm64 - Apple Silicon!), and the build fails if it is missing instead of silently
# producing a backend that cannot launch MCP servers.
COPY --from=docker:27-cli /usr/local/bin/docker /usr/local/bin/docker
RUN docker --version

WORKDIR /srv
COPY backend/pyproject.toml ./
RUN python - <<'PY'
import tomllib, subprocess, sys
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY
COPY backend/app ./app
COPY scripts/fetch_tokenizer.py /tmp/fetch_tokenizer.py
RUN python /tmp/fetch_tokenizer.py /srv/app/llm/assets/tokenizer.json

ENV TOKENIZER_PATH=/srv/app/llm/assets/tokenizer.json
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-access-log"]
