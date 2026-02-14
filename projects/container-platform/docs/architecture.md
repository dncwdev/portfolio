# Architecture

## Why Containerization
- 실행환경을 코드로 정의해 재현성을 확보
- 배포 단위를 이미지로 고정해 환경 차이를 축소
- 런타임 권한/네트워크/리소스 제약을 명시적으로 적용

## Diagram
![Architecture](../assets/arch.png)

## Components
- `app`: 예시 서비스(health/ready/metrics 제공)
- `prometheus`: 메트릭 수집(스크레이핑)
- `grafana`: 대시보드(스냅샷은 SANITIZED)

## Traffic Flow (Request)
1. Client -> `app` HTTP(예: `:8080`)
2. `app` -> stdout logs

## Data Flow (Observability)
1. Prometheus -> `app`의 `/metrics` 스크레이프
2. Grafana -> Prometheus 쿼리로 시각화

## Trust Boundaries / Security Notes
- 외부 노출 포트 최소화(필요한 서비스만 publish)
- 컨테이너는 non-root 실행(권한 최소화)
- 비밀정보는 템플릿만 저장하고 실제 값은 배포 환경에서 주입

