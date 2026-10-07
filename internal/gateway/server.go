package gateway

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/router"
)

// Server handles HTTP API requests for the Sentix Gateway.
type Server struct {
	router    *router.Router
	startTime time.Time
	mux       *http.ServeMux
}

// NewServer initializes HTTP routes, middleware, and Swagger UI.
func NewServer(r *router.Router) *Server {
	mux := http.NewServeMux()
	s := &Server{
		router:    r,
		startTime: time.Now(),
		mux:       mux,
	}

	s.routes()
	return s
}

func (s *Server) routes() {
	// 1. Swagger UI & Documentation
	RegisterSwaggerRoutes(s.mux)

	// 2. Health & Liveness
	s.mux.HandleFunc("GET /health", s.handleHealth)

	// 3. Telemetry & Metrics
	s.mux.HandleFunc("GET /metrics", s.handleMetrics)

	// 4. Primary Sentiment Inference API
	s.mux.HandleFunc("POST /v1/sentiment/analyze", s.handleAnalyze)

	// 5. Reinforcement Learning Feedback Loop API
	s.mux.HandleFunc("POST /v1/sentiment/feedback", s.handleFeedback)
}

// Handler returns the HTTP handler with logging, CORS, and panic recovery middleware applied.
func (s *Server) Handler() http.Handler {
	return s.corsMiddleware(s.recoveryMiddleware(s.loggingMiddleware(s.mux)))
}

func (s *Server) handleAnalyze(w http.ResponseWriter, r *http.Request) {
	// Guard against memory exhaustion with 1MB maximum payload
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var req engine.Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.respondError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON payload: %v", err))
		return
	}

	req.Text = strings.TrimSpace(req.Text)
	if req.Text == "" {
		s.respondError(w, http.StatusBadRequest, "field 'text' cannot be empty")
		return
	}

	if req.Task == "" {
		req.Task = string(engine.TaskDocumentLevel)
	}

	resp, err := s.router.Route(r.Context(), &req)
	if err != nil {
		s.respondError(w, http.StatusInternalServerError, fmt.Sprintf("inference failed: %v", err))
		return
	}

	s.respondJSON(w, http.StatusOK, resp)
}

func (s *Server) handleFeedback(w http.ResponseWriter, r *http.Request) {
	// Guard against memory exhaustion with 1MB maximum payload
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)

	var fb engine.FeedbackRequest
	if err := json.NewDecoder(r.Body).Decode(&fb); err != nil {
		s.respondError(w, http.StatusBadRequest, fmt.Sprintf("invalid JSON payload: %v", err))
		return
	}

	fb.Text = strings.TrimSpace(fb.Text)
	if fb.Text == "" && fb.RequestID == "" {
		s.respondError(w, http.StatusBadRequest, "either 'text' or 'request_id' must be provided")
		return
	}

	if err := s.router.RecordFeedback(&fb); err != nil {
		s.respondError(w, http.StatusInternalServerError, fmt.Sprintf("failed to record feedback: %v", err))
		return
	}

	s.respondJSON(w, http.StatusOK, engine.FeedbackResponse{
		Status:       "success",
		Message:      "reinforcement learning feedback recorded successfully",
		UpdatedStats: s.router.BanditStats(),
	})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	payload := map[string]interface{}{
		"status":         "ok",
		"uptime_seconds": time.Since(s.startTime).Seconds(),
		"engines":        s.router.AvailableEngines(),
	}
	s.respondJSON(w, http.StatusOK, payload)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := s.router.GetMetrics()
	s.respondJSON(w, http.StatusOK, metrics)
}

func (s *Server) respondJSON(w http.ResponseWriter, code int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(payload)
}

func (s *Server) respondError(w http.ResponseWriter, code int, message string) {
	s.respondJSON(w, code, map[string]string{
		"status": "error",
		"error":  message,
	})
}

// Middleware: Logging
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rw := &responseWriterInterceptor{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(rw, r)
		duration := time.Since(start)
		log.Printf("[%s] %s %d - %v", r.Method, r.URL.Path, rw.statusCode, duration)
	})
}

// Middleware: CORS
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// Middleware: Recovery
func (s *Server) recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("[PANIC] %v", rec)
				s.respondError(w, http.StatusInternalServerError, "internal server error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

type responseWriterInterceptor struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriterInterceptor) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}
