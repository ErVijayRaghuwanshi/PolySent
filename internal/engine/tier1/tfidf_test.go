package tier1_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/engine/tier1"
)

func TestTFIDFAnalyzer(t *testing.T) {
	weightsPath := filepath.Join("..", "..", "..", "models", "tier1_tfidf", "weights.json")
	analyzer, err := tier1.NewTFIDFAnalyzer(weightsPath)
	if err != nil {
		t.Fatalf("Failed to initialize TFIDFAnalyzer: %v", err)
	}

	testCases := []struct {
		name          string
		text          string
		expectedClass string
		minConfidence float64
	}{
		{
			name:          "Positive sentiment",
			text:          "The display is crisp and vivid, truly amazing experience!",
			expectedClass: "positive",
			minConfidence: 0.50,
		},
		{
			name:          "Negative sentiment",
			text:          "The customer service was frustratingly slow, completely unhelpful and rude.",
			expectedClass: "negative",
			minConfidence: 0.50,
		},
		{
			name:          "Neutral statement",
			text:          "The package was delivered on Tuesday afternoon via standard carrier.",
			expectedClass: "neutral",
			minConfidence: 0.40,
		},
	}

	ctx := context.Background()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := &engine.Request{
				Text:     tc.text,
				Task:     "document_level",
				Strategy: "ultra_fast",
			}

			start := time.Now()
			resp, err := analyzer.Analyze(ctx, req)
			elapsed := time.Since(start)

			if err != nil {
				t.Fatalf("Analyze failed: %v", err)
			}

			if resp.Overall == nil {
				t.Fatalf("Expected Overall score, got nil")
			}

			if resp.Overall.Label != tc.expectedClass {
				t.Errorf("Expected label %q, got %q (score: %f)", tc.expectedClass, resp.Overall.Label, resp.Overall.Score)
			}

			if resp.Overall.Score < tc.minConfidence {
				t.Errorf("Expected confidence >= %f, got %f", tc.minConfidence, resp.Overall.Score)
			}

			// Sub-millisecond SLA check
			if elapsed > time.Millisecond {
				t.Logf("Notice: execution took %v (> 1ms)", elapsed)
			} else {
				t.Logf("Sub-millisecond SLA verified: %v (latency_ms: %.3f)", elapsed, resp.LatencyMs)
			}
		})
	}
}

func BenchmarkTFIDFAnalyzer(b *testing.B) {
	weightsPath := filepath.Join("..", "..", "..", "models", "tier1_tfidf", "weights.json")
	analyzer, err := tier1.NewTFIDFAnalyzer(weightsPath)
	if err != nil {
		b.Fatalf("Failed to initialize TFIDFAnalyzer: %v", err)
	}

	req := &engine.Request{
		Text:     "The battery life easily lasts two full days with heavy usage.",
		Task:     "document_level",
		Strategy: "ultra_fast",
	}

	ctx := context.Background()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = analyzer.Analyze(ctx, req)
	}
}
