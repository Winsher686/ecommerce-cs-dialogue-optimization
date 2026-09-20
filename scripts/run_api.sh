#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

export USE_MOCK_ENGINE="${USE_MOCK_ENGINE:-0}"

uvicorn src.inference.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
