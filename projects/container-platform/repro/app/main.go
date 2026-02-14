package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync/atomic"
	"time"
)

var (
	ready                 atomic.Bool
	requestsTotal         atomic.Uint64
	requestsHealthTotal   atomic.Uint64
	requestsReadyTotal    atomic.Uint64
	requestsMetricsTotal  atomic.Uint64
	buildVersion          = "dev"
	buildCommit           = "dev"
	buildTime             = "unknown"
)

func main() {
	port := getenvInt("PORT", 8080)
	startupDelay := time.Duration(getenvInt("STARTUP_DELAY_SECONDS", 0)) * time.Second

	ready.Store(false)
	go func() {
		if startupDelay > 0 {
			log.Printf("startup delay: %s", startupDelay)
			time.Sleep(startupDelay)
		}
		ready.Store(true)
		log.Printf("ready: true")
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		requestsTotal.Add(1)
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		fmt.Fprintln(w, "ok")
	})

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		requestsHealthTotal.Add(1)
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		requestsReadyTotal.Add(1)
		if !ready.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})

	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		requestsMetricsTotal.Add(1)
		w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
		fmt.Fprintf(w, "app_build_info{version=%q,commit=%q,build_time=%q} 1\n", buildVersion, buildCommit, buildTime)
		fmt.Fprintf(w, "http_requests_total %d\n", requestsTotal.Load())
		fmt.Fprintf(w, "http_requests_health_total %d\n", requestsHealthTotal.Load())
		fmt.Fprintf(w, "http_requests_ready_total %d\n", requestsReadyTotal.Load())
		fmt.Fprintf(w, "http_requests_metrics_total %d\n", requestsMetricsTotal.Load())
	})

	addr := fmt.Sprintf("0.0.0.0:%d", port)
	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("listening on %s", addr)
	log.Fatal(server.ListenAndServe())
}

func getenvInt(name string, fallback int) int {
	raw, ok := os.LookupEnv(name)
	if !ok || raw == "" {
		return fallback
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return v
}

