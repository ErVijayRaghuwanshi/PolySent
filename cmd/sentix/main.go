package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/ervijay/polysent/internal/engine/tier1"
	"github.com/ervijay/polysent/internal/engine/tier2"
	"github.com/ervijay/polysent/internal/engine/tier3"
	"github.com/ervijay/polysent/internal/gateway"
	"github.com/ervijay/polysent/internal/router"
)

func findDefaultONNXLib() string {
	candidates := []string{
		os.Getenv("ONNXRUNTIME_LIB_PATH"),
		"lib/libonnxruntime.dylib",
		"/opt/homebrew/lib/libonnxruntime.dylib",
		"/usr/local/lib/libonnxruntime.dylib",
	}

	// Also check Python virtual environment dynamic library
	matches, _ := filepath.Glob("python/.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime*.dylib")
	candidates = append(candidates, matches...)

	for _, c := range candidates {
		if c != "" {
			if _, err := os.Stat(c); err == nil {
				abs, err := filepath.Abs(c)
				if err == nil {
					return abs
				}
				return c
			}
		}
	}
	return ""
}

func main() {
	port := flag.Int("port", 8080, "Port for the Sentix Gateway server")
	host := flag.String("host", "0.0.0.0", "Host interface to bind to")
	tier1Weights := flag.String("tier1-weights", "models/tier1_tfidf/weights.json", "Path to Tier-1 JSON weights file")
	tier2Model := flag.String("tier2-model", "models/tier2_distilbert/model_int8.onnx", "Path to Tier-2 INT8 ONNX model")
	tier2Tokenizer := flag.String("tier2-tokenizer", "models/tier2_distilbert/tokenizer.json", "Path to Tier-2 tokenizer.json or vocab.txt")
	tier2Metadata := flag.String("tier2-metadata", "models/tier2_distilbert/metadata.json", "Path to Tier-2 metadata.json")
	onnxLib := flag.String("onnx-lib", "", "Path to libonnxruntime dynamic library")
	poolSize := flag.Int("pool-size", 4, "ONNX Runtime inference session pool size")
	flag.Parse()

	log.Println("==========================================================")
	log.Println("  ⚡ Sentix (PolySent) - Task-Driven Sentiment Gateway   ")
	log.Println("==========================================================")

	// 1. Initialize Tier 1 (Pure Go TF-IDF)
	var t1 *tier1.TFIDFAnalyzer
	if _, err := os.Stat(*tier1Weights); err == nil {
		analyzer, err := tier1.NewTFIDFAnalyzer(*tier1Weights)
		if err != nil {
			log.Printf("[WARN] Failed to load Tier-1 weights: %v", err)
		} else {
			t1 = analyzer
			log.Printf("[INIT] Tier 1 loaded: Pure Go TF-IDF (<1ms SLA)")
		}
	} else {
		log.Printf("[WARN] Tier-1 weights not found at %s", *tier1Weights)
	}

	// 2. Initialize Tier 2 (ONNX Runtime DistilBERT)
	var t2 *tier2.ONNXEngine
	onnxLibPath := *onnxLib
	if onnxLibPath == "" {
		onnxLibPath = findDefaultONNXLib()
	}

	if _, err := os.Stat(*tier2Model); err == nil {
		if onnxLibPath != "" {
			eng, err := tier2.NewONNXEngine(tier2.Config{
				ModelPath:     *tier2Model,
				TokenizerPath: *tier2Tokenizer,
				MetadataPath:  *tier2Metadata,
				SharedLibPath: onnxLibPath,
				PoolSize:      *poolSize,
			})
			if err != nil {
				log.Printf("[WARN] Failed to initialize Tier-2 ONNX engine: %v", err)
			} else {
				t2 = eng
				log.Printf("[INIT] Tier 2 loaded: DistilBERT INT8 ONNX (session pool size: %d, lib: %s)", *poolSize, onnxLibPath)
			}
		} else {
			log.Println("[WARN] libonnxruntime dynamic library not found; Tier 2 disabled")
		}
	} else {
		log.Printf("[WARN] Tier-2 model not found at %s", *tier2Model)
	}

	// 3. Initialize Tier 3 (Local LLM Gateway Adapter)
	t3 := tier3.NewLLMAdapter(tier3.Config{
		BaseURL: "http://localhost:11434",
		Model:   "llama3:8b",
		Timeout: 5 * time.Second,
	})
	log.Printf("[INIT] Tier 3 configured: Local LLM Adapter (fallback mode ready)")

	// 4. Construct Router & Gateway Server
	r := router.NewRouter(t1, t2, t3)
	srv := gateway.NewServer(r)

	addr := fmt.Sprintf("%s:%d", *host, *port)
	httpServer := &http.Server{
		Addr:         addr,
		Handler:      srv.Handler(),
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 15 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// 5. Start HTTP Server in background
	go func() {
		log.Printf("[READY] Gateway listening on http://%s", addr)
		log.Printf("[DOCS]  Swagger UI available at: http://localhost:%d/swagger/", *port)
		log.Printf("[CHECK] Health probe at:         http://localhost:%d/health", *port)
		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[FATAL] HTTP server failure: %v", err)
		}
	}()

	// 6. Graceful Shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("[SHUTDOWN] Shutting down Sentix Gateway gracefully...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("[ERROR] Server shutdown error: %v", err)
	}

	if t2 != nil {
		_ = t2.Close()
	}
	log.Println("[SHUTDOWN] Server exiting cleanly.")
}
