package tier2_test

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/engine/tier2"
)

func TestONNXEngineEndToEnd(t *testing.T) {
	libPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "lib", "libonnxruntime.dylib"))
	modelPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "models", "tier2_distilbert", "model_int8.onnx"))
	tokPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "models", "tier2_distilbert", "tokenizer.json"))
	metaPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "models", "tier2_distilbert", "metadata.json"))

	cfg := tier2.Config{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		MetadataPath:  metaPath,
		SharedLibPath: libPath,
		PoolSize:      2,
	}

	eng, err := tier2.NewONNXEngine(cfg)
	if err != nil {
		t.Fatalf("Failed to create ONNXEngine: %v", err)
	}
	defer eng.Close()

	testQueries := []struct {
		text          string
		expectedClass string
	}{
		{"The display is crisp and vivid, truly amazing experience!", "POSITIVE"},
		{"The customer service was frustratingly slow and completely useless.", "NEGATIVE"},
	}

	ctx := context.Background()
	for _, tc := range testQueries {
		req := &engine.Request{
			Text:     tc.text,
			Task:     "document_level",
			Strategy: "balanced",
		}

		resp, err := eng.Analyze(ctx, req)
		if err != nil {
			t.Fatalf("Analyze failed: %v", err)
		}

		if resp.Overall.Label != tc.expectedClass {
			t.Errorf("Expected label %q, got %q (score: %f)", tc.expectedClass, resp.Overall.Label, resp.Overall.Score)
		}

		t.Logf("Query: %q -> %s (conf: %.4f, latency: %.2fms)", tc.text, resp.Overall.Label, resp.Overall.Score, resp.LatencyMs)
	}
}

func TestONNXEngineConcurrentPool(t *testing.T) {
	libPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "lib", "libonnxruntime.dylib"))
	modelPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "models", "tier2_distilbert", "model_int8.onnx"))
	tokPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "models", "tier2_distilbert", "tokenizer.json"))
	metaPath, _ := filepath.Abs(filepath.Join("..", "..", "..", "models", "tier2_distilbert", "metadata.json"))

	cfg := tier2.Config{
		ModelPath:     modelPath,
		TokenizerPath: tokPath,
		MetadataPath:  metaPath,
		SharedLibPath: libPath,
		PoolSize:      4,
	}

	eng, err := tier2.NewONNXEngine(cfg)
	if err != nil {
		t.Fatalf("Failed to create ONNXEngine: %v", err)
	}
	defer eng.Close()

	var wg sync.WaitGroup
	concurrentReqs := 8
	ctx := context.Background()

	start := time.Now()
	for i := 0; i < concurrentReqs; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			req := &engine.Request{
				Text:     "Remarkable speed and pristine build quality!",
				Task:     "document_level",
				Strategy: "balanced",
			}
			resp, err := eng.Analyze(ctx, req)
			if err != nil {
				t.Errorf("Worker %d failed: %v", id, err)
				return
			}
			if resp.Overall.Label != "POSITIVE" {
				t.Errorf("Worker %d expected POSITIVE, got %s", id, resp.Overall.Label)
			}
		}(i)
	}
	wg.Wait()
	totalDuration := time.Since(start)
	t.Logf("Completed %d concurrent inferences in %v (avg: %v/req)", concurrentReqs, totalDuration, totalDuration/time.Duration(concurrentReqs))
}
