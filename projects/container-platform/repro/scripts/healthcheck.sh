#!/usr/bin/env sh
set -eu

url="${1:-http://localhost:8080/readyz}"

attempts="${ATTEMPTS:-30}"
sleep_seconds="${SLEEP_SECONDS:-1}"

echo "checking readiness: ${url}"

i=1
while [ "$i" -le "$attempts" ]; do
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "ready"
      exit 0
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -q -O - "$url" >/dev/null 2>&1; then
      echo "ready"
      exit 0
    fi
  else
    echo "error: curl or wget is required" >&2
    exit 2
  fi

  sleep "$sleep_seconds"
  i=$((i + 1))
done

echo "not ready after ${attempts} attempts" >&2
exit 1

