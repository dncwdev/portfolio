# Performance

## What to measure
- 이미지 크기(빌드 산출물 최소화)
- cold start / readiness 시간
- 리소스 사용량(CPU/메모리)

## How to capture evidence
- `docker images`로 크기 비교
- 요청/메트릭 기반으로 초기화 시간 기록
- (선택) 간단한 부하 테스트 결과 첨부(SANITIZED)

