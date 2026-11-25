# Build context: repository root  (docker build -f backend/Dockerfile .)
FROM docker.arvancloud.ir/python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# NOTE: no docker CLI is needed: the backend talks to the Docker daemon through the Python SDK
# (sandbox lifecycle, exec and MCP-over-exec streams).

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
