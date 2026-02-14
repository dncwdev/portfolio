# syntax=docker/dockerfile:1

FROM golang:1.22-alpine AS build
WORKDIR /src

# Keep the build reproducible and small.
ENV CGO_ENABLED=0

COPY app/go.mod ./app/go.mod
WORKDIR /src/app
RUN go mod download

COPY app/. ./
RUN go build -trimpath -ldflags="-s -w" -o /out/app .

FROM alpine:3.20
WORKDIR /

RUN addgroup -S app && adduser -S -G app -H -s /sbin/nologin app

ENV PORT=8080

COPY --from=build /out/app /app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /app

USER app
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=2s --retries=3 CMD wget -q -O - "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 || exit 1

ENTRYPOINT ["/entrypoint.sh"]

