#!/usr/bin/env sh
set -eu

script_dir="$(CDPATH= cd -- "$(dirname "$0")" && pwd)"
repro_dir="$(CDPATH= cd -- "${script_dir}/.." && pwd)"
compose_dir="${repro_dir}/compose"
env_file="${compose_dir}/.env"

if [ ! -f "$env_file" ]; then
  cp "${compose_dir}/.env.template" "$env_file"
fi

# shellcheck disable=SC1090
set -a
. "$env_file"
set +a

docker compose -f "${compose_dir}/compose.dev.yml" --env-file "$env_file" up -d --build

"${script_dir}/healthcheck.sh" "http://localhost:${APP_PORT}/readyz"

echo "smoke test passed"

