# Container Platform (Curated Case Study)

Ops-focused case study for standardizing a Docker/Compose runtime for internal services.
Includes a sanitized minimal reproduction (a small Go service + Compose stack) and runbook-style docs.

## TL;DR

- **What:** Docker/Compose-based runtime baseline for repeatable builds, health checks, and observability.
- **My role:** Designed and implemented the baseline; authored operational docs and troubleshooting guides.
- **Outcome:** More predictable deployments, faster onboarding, and improved debuggability (qualitative).

---

## Problem and constraints

- Restricted / regulated enterprise environment (limited internet access, controlled registries).
- Need repeatable operations with a small team: documented start/update/rollback procedures.
- Must publish a public portfolio without disclosing proprietary systems: sanitize and provide a minimal repro.

---

## Architecture

_Architecture diagram: coming soon (will be added once the case study is fully sanitized)._

### Key components (high level)

- Runtime: Docker and Docker Compose (lightweight baseline; Kubernetes-friendly patterns where applicable).
- Image build: multi-stage builds, pinned versions, non-root execution.
- Config and secrets: templates only in repo; runtime injection per environment.
- Observability: Prometheus-style metrics and Grafana dashboards (sanitized).
- Operations: runbook-driven deploy/update/rollback and smoke tests.

### Design notes (WIP)

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/decisions.md`](docs/decisions.md)
- [`docs/security.md`](docs/security.md)
- [`docs/performance.md`](docs/performance.md)

---

## Minimal reproduction

This repo includes a sanitized minimal reproduction:

- App: [`repro/app/`](repro/app/) (Go service with `/healthz`, `/readyz`, `/metrics`)
- Dockerfile: [`repro/docker/`](repro/docker/)
- Compose: [`repro/compose/`](repro/compose/)
- Scripts: [`repro/scripts/`](repro/scripts/)

Example:

```bash
cp projects/container-platform/repro/compose/.env.template projects/container-platform/repro/compose/.env
docker compose -f projects/container-platform/repro/compose/compose.dev.yml --env-file projects/container-platform/repro/compose/.env up -d --build
bash projects/container-platform/repro/scripts/smoke_test.sh
```

---

## Operational docs (WIP)

- Runbook: [`docs/operations.md`](docs/operations.md)
- Troubleshooting: [`docs/troubleshooting.md`](docs/troubleshooting.md)
- Monitoring snapshot: `assets/grafana.png` (sanitized placeholder)

---

## Scope / non-goals

**In scope**

- Container image and runtime hardening (non-root, health/readiness, predictable builds).
- Minimal observability (metrics + dashboards).
- Operations documentation for repeatable workflows.

**Out of scope**

- Proprietary business logic and internal configurations.
- Vendor-specific details that would identify the organization.
- Full production deployment manifests for internal systems.

---

## Confidentiality

- All screenshots, logs, and identifiers are sanitized.
- Secrets/tokens/certs are not included (templates only).

