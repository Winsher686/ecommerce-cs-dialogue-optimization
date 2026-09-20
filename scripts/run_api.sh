#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export USE_MOCK_ENGINE="${USE_MOCK_ENGINE:-0}"
export PORT="${PORT:-8000}"

python -m src.inference.api
