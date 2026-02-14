# Operations (Runbook)

## Start (Dev)
```bash
cp projects/container-platform/repro/compose/.env.template projects/container-platform/repro/compose/.env
docker compose -f projects/container-platform/repro/compose/compose.dev.yml --env-file projects/container-platform/repro/compose/.env up -d --build
```

## Stop
```bash
docker compose -f projects/container-platform/repro/compose/compose.dev.yml --env-file projects/container-platform/repro/compose/.env down
```

## Update
```bash
docker compose -f projects/container-platform/repro/compose/compose.dev.yml --env-file projects/container-platform/repro/compose/.env up -d --build
```

## Rollback (Example)
- 이전 이미지 태그로 되돌린 뒤 `up -d` 수행
- 실제 운영에서는 이미지 태그/릴리즈 노트/변경 이력 기준으로 수행

## Backup (Example)
- Prometheus 데이터(볼륨)를 백업 대상으로 포함 가능
- 실제 운영에서는 백업 주기/보관 정책/복구 리허설을 문서화

## Logs
```bash
docker compose -f projects/container-platform/repro/compose/compose.dev.yml --env-file projects/container-platform/repro/compose/.env logs -f --tail=200
```

