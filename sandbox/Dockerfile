# Per-session agent computer. One container of this image is created for every chat session.
# Build context is the repository root (so the web helper can be copied in):
#   docker build -f sandbox/Dockerfile -t decillion-sandbox:latest .
FROM docker.arvancloud.ir/python:3.12-slim-bookworm

ARG NODE_MAJOR=22
ARG INSTALL_PLAYWRIGHT=false
ARG PY_PACKAGES="numpy pandas matplotlib scipy requests beautifulsoup4 html2text pytest pyyaml"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    UV_TOOL_DIR=/opt/uv-tools \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never \
    PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers \
    NPM_CONFIG_UPDATE_NOTIFIER=false

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g' /etc/apt/sources.list.d/debian.sources \
    && sed -i 's|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
      git ripgrep jq curl wget ca-certificates gnupg procps psmisc \
      build-essential cmake make pkg-config unzip zip tree sqlite3 openssh-client less nano vim-tiny \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python toolchain + data-science basics the agent commonly needs
RUN pip install uv ${PY_PACKAGES}

# Popular MCP servers, pre-installed so enabling one in the UI never needs a download.
# (Python servers via uv tools, Node servers globally; each runs over stdio via `docker exec -i`.)
RUN uv tool install mcp-server-time \
 && uv tool install mcp-server-git \
 && uv tool install mcp-server-fetch \
 && uv tool install mcp-server-sqlite
RUN npm install -g --no-fund --no-audit \
      @modelcontextprotocol/server-filesystem \
      @modelcontextprotocol/server-memory \
      @modelcontextprotocol/server-sequential-thinking \
      @modelcontextprotocol/server-github \
      @modelcontextprotocol/server-brave-search \
      @upstash/context7-mcp \
      @playwright/mcp \
 && npm cache clean --force
# Fail the build (not the first chat) if any MCP server binary is missing or not runnable by the agent user.
RUN for b in mcp-server-time mcp-server-git mcp-server-fetch mcp-server-sqlite mcp-server-filesystem mcp-server-memory \
             mcp-server-sequential-thinking mcp-server-github mcp-server-brave-search context7-mcp playwright-mcp; do \
      command -v "$b" >/dev/null || { echo "MISSING MCP BINARY: $b" >&2; exit 1; }; \
    done

RUN if [ "$INSTALL_PLAYWRIGHT" = "true" ]; then npx --yes playwright@latest install --with-deps chromium; fi

# Sandboxed web access helper (SSRF-guarded fetch) used by the web_fetch tool
COPY backend/app/tools/web_impl.py /usr/local/lib/a1_web.py

RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
 && mkdir -p /workspace /tmp/a1jobs \
 && chown -R agent:agent /workspace /tmp/a1jobs /home/agent \
 && git config --system user.name "agent" && git config --system user.email "agent@sandbox.local" \
 && git config --system init.defaultBranch main && git config --system safe.directory '*'

# The agent user must be able to *run* the python MCP servers (their venv interpreter must not live under /root).
RUN su agent -c 'for t in time git fetch sqlite; do py=$(readlink -f /opt/uv-tools/mcp-server-$t/bin/python) && test -x "$py" || { echo "mcp-server-$t interpreter not usable by agent: $py" >&2; exit 1; }; done'

USER agent
WORKDIR /workspace
ENV HOME=/home/agent PATH=/home/agent/.local/bin:$PATH
CMD ["sleep", "infinity"]
