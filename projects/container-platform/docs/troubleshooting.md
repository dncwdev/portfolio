# Troubleshooting

## Port already in use
- `APP_PORT`/`PROMETHEUS_PORT`/`GRAFANA_PORT` 값을 변경
- 사용 중인 프로세스 확인 후 충돌 해소

## Healthcheck failing
- `app` 로그 확인: `docker compose logs app`
- `/healthz`, `/readyz` 엔드포인트 직접 호출

## Prometheus not scraping
- Prometheus 타겟 상태 확인(웹 UI Targets)
- `prometheus.yml`의 `targets`/포트/네트워크 확인

