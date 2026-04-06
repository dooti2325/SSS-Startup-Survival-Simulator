#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${1:-$(pwd)}"
DOCKER_BUILD_TIMEOUT="${DOCKER_BUILD_TIMEOUT:-900}"

BOLD="$(printf '\033[1m')"
GREEN="$(printf '\033[32m')"
RED="$(printf '\033[31m')"
YELLOW="$(printf '\033[33m')"
NC="$(printf '\033[0m')"

log() {
  printf "%s\n" "$1"
}

pass() {
  printf "%s[PASS]%s %s\n" "$GREEN" "$NC" "$1"
}

fail() {
  printf "%s[FAIL]%s %s\n" "$RED" "$NC" "$1"
}

hint() {
  printf "%sHint:%s %s\n" "$YELLOW" "$NC" "$1"
}

stop_at() {
  printf "%sStopped at:%s %s\n" "$YELLOW" "$NC" "$1"
  exit 1
}

run_with_timeout() {
  local timeout_seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout "$timeout_seconds" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then
    gtimeout "$timeout_seconds" "$@"
  else
    "$@"
  fi
}

log "${BOLD}Step 1/3: Checking repo contents${NC} ..."

for required_file in inference.py openenv.yaml requirements.txt; do
  if [ ! -f "$REPO_DIR/$required_file" ]; then
    fail "Missing required file: $required_file"
    stop_at "Step 1"
  fi
done

pass "Required repo files are present"

log "${BOLD}Step 2/3: Running docker build${NC} ..."

if ! command -v docker >/dev/null 2>&1; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT="$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1)" && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running openenv validate${NC} ..."

if ! command -v openenv >/dev/null 2>&1; then
  fail "openenv command not found"
  hint "Install it: pip install openenv-core"
  stop_at "Step 3"
fi

VALIDATE_OK=false
VALIDATE_OUTPUT="$(cd "$REPO_DIR" && openenv validate 2>&1)" && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 3/3 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0
