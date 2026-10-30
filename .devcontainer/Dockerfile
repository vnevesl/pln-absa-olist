# Container do Ralph: Python (CPU) + Node (controlador) + Claude Code. Sem GPU, sem segredos na imagem.
FROM docker.io/library/node:24-bookworm-slim

ARG CLAUDE_CODE_VERSION=2.1.258

RUN apt-get update \
  && apt-get install --yes --no-install-recommends ca-certificates curl git jq procps ripgrep sudo \
       python3 python3-venv python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && npm install --global "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
  && printf 'node ALL=(root) NOPASSWD:ALL\n' > /etc/sudoers.d/node && chmod 0440 /etc/sudoers.d/node

# Dependências Python do projeto (CPU). O pacote em si é instalado em modo editável no início da sessão.
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv /opt/venv \
  && pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir "pandas>=2" "numpy>=1.24" "scikit-learn>=1.3" "scipy>=1.10" "unidecode>=1.3" \
       "transformers>=4.44" "accelerate>=0.30" "pytest>=8" "ruff>=0.6" \
  && chown -R node:node /opt/venv

RUN git config --system --add safe.directory '*' \
  && mkdir -p /home/node/.claude && chown -R node:node /home/node/.claude

USER node
ENV CLAUDE_CONFIG_DIR=/home/node/.claude
WORKDIR /workspaces/pln-absa-olist
