package router_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/engine/tier1"
	"github.com/ervijay/polysent/internal/engine/tier2"
	"github.com/ervijay/polysent/internal/engine/tier3"
	"github.com/ervijay/polysent/internal/router"
)

func TestRouterStrategies(t *testing.T) {
	weightsPath, _ := filepath.Abs(filepath.Join("..", "..", "models", "tier1_tfidf", "weights.json"))
	t1, err := tier1.NewTFIDFAnalyzer(weightsPath)
	if err != nil {
		t.Fatalf("Failed to initialize Tier-1: %v", err)
	}

	libPath, _ := filepath.Abs(filepath.Join("..", "..", "lib", "libonnxruntime.dylib"))
	modelPath, _ := filepath.Abs(filepath.Join("..", "..", "models", "tier2_distilbert", "model_int8.onnx"))
	tokPath, _ := filepath.Abs(filepath.Join("..", "..", "models", "tier2_distilbert", "tokenizer.json"))
	metaPath, _ := filepath.Abs(filepath.Join("..", "..", "models", "tier2_distilbert", "metadata.json"))

	t2, err := tier2.NewONNXEngine(tier2.Config{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		MetadataPath:  metaPath,
		SharedLibPath: libPath,
		PoolSize:      1,
	})
	if err != nil {
		t.Fatalf("Failed to initialize Tier-2: %v", err)
	}
	defer t2.Close()

	t3 := tier3.NewLLMAdapter(tier3.Config{
		BaseURL: "http://localhost:11434",
	})

	r := router.NewRouter(t1, t2, t3)
	ctx := context.Background()

	// 1. Ultra Fast -> Tier 1
	resp1, err := r.Route(ctx, &engine.Request{
		Text:     "Awesome phone!",
		Task:     "document_level",
		Strategy: "ultra_fast",
	})
	if err != nil {
		t.Fatalf("Ultra-fast route failed: %v", err)
	}
	if resp1.EngineUsed != t1.Name() {
		t.Errorf("Expected engine %s, got %s", t1.Name(), resp1.EngineUsed)
	}

	// 2. Balanced -> Tier 2
	resp2, err := r.Route(ctx, &engine.Request{
		Text:     "The display is crisp and vivid, truly amazing experience!",
		Task:     "document_level",
		Strategy: "balanced",
	})
	if err != nil {
		t.Fatalf("Balanced route failed: %v", err)
	}
	if resp2.EngineUsed != t2.Name() {
		t.Errorf("Expected engine %s, got %s", t2.Name(), resp2.EngineUsed)
	}

	// 3. Fallback when Tier-2 is missing
	rNoTier2 := router.NewRouter(t1, nil, nil)
	respFallback, err := rNoTier2.Route(ctx, &engine.Request{
		Text:     "Great device",
		Strategy: "balanced",
	})
	if err != nil {
		t.Fatalf("Fallback route failed: %v", err)
	}
	if respFallback.EngineUsed != t1.Name() {
		t.Errorf("Expected fallback to Tier-1, got %s", respFallback.EngineUsed)
	}

	// 4. Check Metrics
	m := r.GetMetrics()
	if m.TotalRequests < 2 {
		t.Errorf("Expected >= 2 requests in metrics, got %d", m.TotalRequests)
	}
	if m.Tier1Hits < 1 || m.Tier2Hits < 1 {
		t.Errorf("Expected tier hits recorded, got %+v", m)
	}
}
