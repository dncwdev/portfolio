# Security

## Non-root
- 런타임 컨테이너는 non-root 사용자로 실행
- 필요 시 `read_only`, `cap_drop`, seccomp/apparmor 등 강화 가능(환경별)

## Secrets & Config
- 저장소에는 `.env.template`만 포함
- 실제 값은 CI/CD 또는 런타임 시크릿 스토어에서 주입

## Image Hygiene
- 베이스 이미지 버전 핀/정기 업데이트
- 취약점 스캐닝(예: Trivy/Grype)과 SBOM 생성(예: Syft)을 파이프라인에 포함 권장

## Logging
- 민감정보(토큰/개인정보) 로그 출력 금지
- 대시보드/스크린샷은 SANITIZED 버전만 공유

