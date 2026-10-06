package router_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/router"
)

func TestContextualBanditLearning(t *testing.T) {
	tempLog := filepath.Join(t.TempDir(), "test_feedback.jsonl")
	bandit := router.NewContextualBandit(tempLog)

	// 1. Initial selection
	reqShort := &engine.Request{Text: "Great!"}
	stratShort := bandit.SelectArm(reqShort)
	if stratShort != engine.StrategyUltraFast {
		t.Errorf("Expected ultra_fast for short text, got %s", stratShort)
	}

	reqLong := &engine.Request{Text: "The display is crisp and vivid, truly amazing experience with stellar speed!"}
	stratLong := bandit.SelectArm(reqLong)
	if stratLong != engine.StrategyBalanced {
		t.Errorf("Expected balanced for standard text, got %s", stratLong)
	}

	// 2. Record Positive Reward for Tier 1
	err := bandit.RecordReward(&engine.FeedbackRequest{
		RequestID:      "test-1",
		Text:           "Great!",
		EngineUsed:     "tier1_purego_tfidf_logreg",
		PredictedLabel: "positive",
		Reward:         1.0,
		LatencyMs:      0.05,
	})
	if err != nil {
		t.Fatalf("Failed to record reward: %v", err)
	}

	// 3. Record Negative Reward for Tier 2
	err = bandit.RecordReward(&engine.FeedbackRequest{
		RequestID:      "test-2",
		Text:           "Average battery life.",
		EngineUsed:     "onnx_distilbert_int8",
		PredictedLabel: "NEGATIVE",
		CorrectLabel:   "neutral",
		Reward:         -1.0,
		LatencyMs:      15.0,
	})
	if err != nil {
		t.Fatalf("Failed to record reward: %v", err)
	}

	// 4. Verify Log File Written
	data, err := os.ReadFile(tempLog)
	if err != nil {
		t.Fatalf("Failed to read feedback log: %v", err)
	}
	if len(data) == 0 {
		t.Errorf("Expected non-empty feedback log file")
	}

	// 5. Verify Stats
	stats := bandit.Stats()
	if stats["total_feedback_pulls"].(int64) != 2 {
		t.Errorf("Expected 2 total feedback pulls, got %v", stats["total_feedback_pulls"])
	}

	t.Logf("Bandit learning verified: %+v", stats)
}
