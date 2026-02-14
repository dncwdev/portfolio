#!/bin/sh
set -eu

: "${PORT:=8080}"
: "${STARTUP_DELAY_SECONDS:=0}"

echo "starting app (PORT=${PORT}, STARTUP_DELAY_SECONDS=${STARTUP_DELAY_SECONDS})"

exec /app

