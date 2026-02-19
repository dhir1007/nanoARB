#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
UI_DIR="$ROOT/nano-arb-ui-development"

cleanup() {
  echo ""
  echo "Shutting down..."
  kill "$BACKEND_PID" 2>/dev/null || true
  kill "$UI_PID" 2>/dev/null || true
  wait "$BACKEND_PID" 2>/dev/null || true
  wait "$UI_PID" 2>/dev/null || true
  echo "All processes stopped."
}
trap cleanup EXIT INT TERM

# Kill anything already on our ports
lsof -ti:9090 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo "============================================"
echo "  NanoARB - Starting Full Stack"
echo "============================================"

# --- Build Rust backend ---
echo ""
echo "[1/3] Building Rust trading engine (release)..."
cargo build --release --bin nanoarb --manifest-path "$ROOT/Cargo.toml"

# --- Install UI deps if needed ---
if [ -d "$UI_DIR" ]; then
  if [ ! -d "$UI_DIR/node_modules" ]; then
    echo ""
    echo "[2/3] Installing UI dependencies..."
    (cd "$UI_DIR" && pnpm install)
  else
    echo ""
    echo "[2/3] UI dependencies already installed."
  fi
else
  echo ""
  echo "[2/3] UI directory not found at $UI_DIR - skipping frontend."
fi

# --- Start backend ---
echo ""
echo "[3/3] Starting services..."
echo ""

"$ROOT/target/release/nanoarb" &
BACKEND_PID=$!
sleep 2

if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  echo "ERROR: Backend failed to start."
  exit 1
fi

echo "  Rust Engine   : http://localhost:9090  (PID $BACKEND_PID)"
echo "  Metrics       : http://localhost:9090/metrics"
echo "  Health        : http://localhost:9090/health"
echo "  API State     : http://localhost:9090/api/state"
echo "  SSE Stream    : http://localhost:9090/api/stream"

# --- Start UI ---
if [ -d "$UI_DIR/node_modules" ]; then
  (cd "$UI_DIR" && pnpm dev --port 3000) &
  UI_PID=$!
  sleep 5
  echo "  Dashboard UI  : http://localhost:3000   (PID $UI_PID)"
else
  UI_PID=0
  echo "  Dashboard UI  : (not available - see README)"
fi

echo ""
echo "============================================"
echo "  All services running. Press Ctrl+C to stop."
echo "============================================"
echo ""

wait
