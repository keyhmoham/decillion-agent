# Build context: repository root  (docker build -f backend/Dockerfile .)
FROM python:3.12-slim-bookworm

ARG DOCKER_CLI_VERSION=27.3.1
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    # docker CLI only: MCP servers are started inside the sandbox with `docker exec -i`
    && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
       | tar -xz --strip-components=1 -C /usr/local/bin docker/docker \
    || echo "WARNING: docker CLI not installed (MCP-in-sandbox will be unavailable)"

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
