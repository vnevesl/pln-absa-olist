#!/usr/bin/env bash
# Sobe o container do Ralph com o repositório montado. Uso:
#   scripts/ralph/container.sh            # shell interativo
#   scripts/ralph/container.sh <comando>  # roda um comando e sai
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
docker build --quiet -t pln-absa-ralph -f .devcontainer/Dockerfile .devcontainer >/dev/null
# O login do Claude fica num volume do Docker (nunca no repositório).
exec docker run --rm -it --init \
  -v "$PWD":/workspaces/pln-absa-olist \
  -v pln-absa-ralph-claude:/home/node/.claude \
  -e CLAUDE_CONFIG_DIR=/home/node/.claude \
  pln-absa-ralph bash -lc 'pip install --quiet --no-deps -e . && cd scripts/ralph && npm install --silent && cd ../.. && exec "${@:-bash}"' _ "$@"
