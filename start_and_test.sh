#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    VENV_DIR="$PROJECT_DIR/.venv"
elif [[ -x "$PROJECT_DIR/../.venv/bin/python" ]]; then
    VENV_DIR="$PROJECT_DIR/../.venv"
else
    echo "ERROR: No virtual environment found." >&2
    echo "Create one with: python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
    exit 1
fi

PYTHON="$VENV_DIR/bin/python"
UVICORN="$VENV_DIR/bin/uvicorn"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:$PORT"
SERVER_LOG="$(mktemp /tmp/ai_a2_2527953_uvicorn.XXXXXX.log)"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$SERVER_LOG"
}
trap cleanup EXIT INT TERM

echo "== Environment =="
echo "Project: $PROJECT_DIR"
"$PYTHON" --version

echo
echo "== Reproducible build =="
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ai_a2_2527953_matplotlib}" \
    "$PYTHON" run_all.py --student-id ug2527953@ines.ac.rw --seed 7953

echo
echo "== Valid command-line prediction =="
"$PYTHON" predict.py --json samples/valid.json

echo
echo "== Invalid command-line prediction (failure expected) =="
set +e
"$PYTHON" predict.py --json samples/invalid.json
INVALID_STATUS=$?
set -e
if [[ "$INVALID_STATUS" -eq 0 ]]; then
    echo "ERROR: invalid input was unexpectedly accepted" >&2
    exit 1
fi
echo "PASS: invalid input was rejected with exit code $INVALID_STATUS"

echo
echo "== Automated tests =="
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ai_a2_2527953_matplotlib}" \
    "$PYTHON" -m pytest -q

echo
echo "== Starting FastAPI on $BASE_URL =="
"$UVICORN" app:app --host 127.0.0.1 --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

READY=0
for _ in {1..30}; do
    if curl --silent --fail "$BASE_URL/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: FastAPI stopped during startup" >&2
        sed -n '1,120p' "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 0.25
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: FastAPI did not become ready" >&2
    sed -n '1,120p' "$SERVER_LOG" >&2
    exit 1
fi

echo
echo "GET /health"
curl --silent --show-error --fail "$BASE_URL/health"
echo

echo
echo "GET /model-info"
curl --silent --show-error --fail "$BASE_URL/model-info"
echo

echo
echo "POST /predict"
curl --silent --show-error --fail \
    --request POST "$BASE_URL/predict" \
    --header 'Content-Type: application/json' \
    --data '{"rainfall_mm":950.0,"avg_temp_c":18.0,"soil_moisture_pct":55.0,"fertilizer_kg":90.0,"field_size_ha":1.5,"season":"A","cooperative_support":1}'
echo

echo
echo "POST /predict with invalid rainfall (HTTP 422 expected)"
HTTP_STATUS="$(curl --silent --output /tmp/ai_a2_2527953_invalid_response.json --write-out '%{http_code}' \
    --request POST "$BASE_URL/predict" \
    --header 'Content-Type: application/json' \
    --data '{"rainfall_mm":2000.0}')"
if [[ "$HTTP_STATUS" != "422" ]]; then
    echo "ERROR: expected HTTP 422, received $HTTP_STATUS" >&2
    exit 1
fi
echo "PASS: invalid API input returned HTTP $HTTP_STATUS"
rm -f /tmp/ai_a2_2527953_invalid_response.json

echo
echo "ALL AUTOMATED BUILD AND TEST CHECKS PASSED"
if [[ "${KEEP_SERVER:-0}" == "1" ]]; then
    echo "Swagger documentation: $BASE_URL/docs"
    echo "The API is still running. Press Ctrl+C to stop it."
    wait "$SERVER_PID"
else
    echo "Swagger documentation: $BASE_URL/docs (server stops when this script exits)"
    echo "Use KEEP_SERVER=1 ./start_and_test.sh to keep the API running after the tests."
fi

