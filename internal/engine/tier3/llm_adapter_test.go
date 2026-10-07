package tier3_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/engine/tier3"
)

func TestLLMAdapterSuccess(t *testing.T) {
	// Mock OpenAI/vLLM completion server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("Unexpected URL path: %s", r.URL.Path)
		}

		resp := map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]string{
						"content": "{\"label\": \"positive\", \"score\": 0.98}",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Test URL with trailing /v1 to ensure no duplicate /v1/v1
	adapter := tier3.NewLLMAdapter(tier3.Config{
		BaseURL: server.URL + "/v1",
		Model:   "llama3:8b",
		Timeout: 2 * time.Second,
	})

	req := &engine.Request{
		Text:    "The battery life is exceptional!",
		Task:    "aspect_based",
		Aspects: []string{"battery"},
	}

	resp, err := adapter.Analyze(context.Background(), req)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	if resp.Overall.Label != "positive" {
		t.Errorf("Expected label positive, got %s", resp.Overall.Label)
	}
	if resp.Overall.Score != 0.98 {
		t.Errorf("Expected score 0.98, got %f", resp.Overall.Score)
	}
	if len(resp.Aspects) != 1 || resp.Aspects[0].Aspect != "battery" {
		t.Errorf("Expected aspect battery, got %+v", resp.Aspects)
	}
}

func TestLLMAdapterMarkdownCodeBlockParsing(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]string{
						"content": "```json\n{\"label\": \"negative\", \"score\": 0.91}\n```",
					},
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	adapter := tier3.NewLLMAdapter(tier3.Config{
		BaseURL: server.URL,
		Model:   "llama3:8b",
		Timeout: 2 * time.Second,
	})

	req := &engine.Request{
		Text: "Terrible quality, broke immediately.",
		Task: "document_level",
	}

	resp, err := adapter.Analyze(context.Background(), req)
	if err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}

	if resp.Overall.Label != "negative" {
		t.Errorf("Expected label negative, got %s", resp.Overall.Label)
	}
}

func TestLLMAdapterOfflineFallback(t *testing.T) {
	// Point to an unavailable port
	adapter := tier3.NewLLMAdapter(tier3.Config{
		BaseURL: "http://127.0.0.1:59999",
		Model:   "llama3:8b",
		Timeout: 100 * time.Millisecond,
	})

	req := &engine.Request{
		Text: "Ambiguous sarcastic phrase.",
		Task: "document_level",
	}

	resp, err := adapter.Analyze(context.Background(), req)
	if err != nil {
		t.Fatalf("Expected graceful fallback, got error: %v", err)
	}

	if resp.EngineUsed != "tier3_llm_gateway_fallback" {
		t.Errorf("Expected fallback engine name, got %s", resp.EngineUsed)
	}
	if resp.Overall.Label != "mixed" {
		t.Errorf("Expected mixed label on fallback, got %s", resp.Overall.Label)
	}
}
