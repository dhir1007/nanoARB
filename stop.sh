#!/usr/bin/env bash
set -euo pipefail

echo "Stopping NanoARB services..."

lsof -ti:9090 | xargs kill -9 2>/dev/null && echo "  Rust engine stopped." || echo "  Rust engine not running."
lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "  UI dev server stopped." || echo "  UI dev server not running."

# Stop monitoring stack if running
if command -v docker &>/dev/null; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  if [ -f "$SCRIPT_DIR/docker/docker-compose-monitoring.yml" ]; then
    docker compose -f "$SCRIPT_DIR/docker/docker-compose-monitoring.yml" down 2>/dev/null && echo "  Monitoring stack stopped." || true
  fi
fi

echo "Done."
