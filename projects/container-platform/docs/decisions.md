# Decisions

## Constraints (Examples)
- 폐쇄망/망분리 환경: 외부 의존 최소화, 내부 레지스트리/캐시 전략 필요
- 보안/감사: non-root, 최소 권한, 로그/설정 관리 기준 준수
- 운영 인력 제약: runbook과 자동화로 반복 작업 최소화

## Key Choices

### Base Image
- 빌드: `golang:*-alpine` (예시)
- 런타임: `alpine:*` (shell 기반 entrypoint/healthcheck 용이)
- 실제 운영에서는 버전 핀 + 주기적 업데이트/스캐닝을 전제로 함

### Multi-stage Build
- 컴파일/빌드 툴을 런타임에서 제거해 이미지 크기와 공격면을 축소

### Non-root
- 컨테이너 내부 사용자 생성 후 `USER` 지정
- 호스트 바인드 포트는 1024 이상 사용(예: 8080)

### Healthcheck & Readiness
- Dockerfile `HEALTHCHECK`로 기본 생존 체크 제공
- readiness는 앱 엔드포인트(`/readyz`)로 별도 제공(초기화/의존성 반영 가능)

### Observability
- 로그: stdout/stderr
- 메트릭: `/metrics`(Prometheus 텍스트 포맷) + Prometheus/Grafana(선택)

## What’s intentionally omitted (confidential)
- 내부 IP/도메인/호스트명/계정/토큰/인증서
- 실제 `.env`, 운영 로그 원문, 재배포 제한 파일

