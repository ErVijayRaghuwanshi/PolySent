package router

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ervijay/polysent/internal/engine"
)

// ArmStats stores the empirical reward and selection statistics for each serving tier.
type ArmStats struct {
	Name           string  `json:"name"`
	Tier           int     `json:"tier"`
	PullCount      int64   `json:"pull_count"`
	TotalReward    float64 `json:"total_reward"`
	AverageReward  float64 `json:"average_reward"`
	AvgLatencyMs   float64 `json:"avg_latency_ms"`
}

// ExperienceRecord logs feedback data to disk for offline RL optimization and replay.
type ExperienceRecord struct {
	Timestamp      string  `json:"timestamp"`
	RequestID      string  `json:"request_id"`
	Text           string  `json:"text"`
	EngineUsed     string  `json:"engine_used"`
	PredictedLabel string  `json:"predicted_label"`
	CorrectLabel   string  `json:"correct_label"`
	Reward         float64 `json:"reward"`
	LatencyMs      float64 `json:"latency_ms"`
	Comment        string  `json:"comment,omitempty"`
}

// ContextualBandit orchestrates exploration vs exploitation using Upper Confidence Bound (UCB1).
type ContextualBandit struct {
	mu           sync.RWMutex
	arms         map[string]*ArmStats
	totalPulls   int64
	cExploration float64 // Exploration hyperparameter (typically sqrt(2))
	logPath      string
}

// NewContextualBandit initializes arms and optional persistent log buffer.
func NewContextualBandit(logPath string) *ContextualBandit {
	if logPath == "" {
		logPath = "data/rl_feedback.jsonl"
	}

	arms := map[string]*ArmStats{
		"tier1": {Name: "tier1_purego_tfidf_logreg", Tier: 1, AverageReward: 0.70, AvgLatencyMs: 0.05},
		"tier2": {Name: "onnx_distilbert_int8", Tier: 2, AverageReward: 0.90, AvgLatencyMs: 8.0},
		"tier3": {Name: "tier3_llm_gateway", Tier: 3, AverageReward: 0.95, AvgLatencyMs: 250.0},
	}

	return &ContextualBandit{
		arms:         arms,
		cExploration: 1.414,
		logPath:      logPath,
	}
}

// SelectArm picks the optimal tier based on SLA constraints and UCB1 score.
func (b *ContextualBandit) SelectArm(req *engine.Request) engine.Strategy {
	b.mu.RLock()
	defer b.mu.RUnlock()

	// 1. Explicit SLA constraints take strict priority
	strategy := engine.Strategy(req.Strategy)
	if strategy == engine.StrategyUltraFast || strategy == engine.StrategyBalanced || strategy == engine.StrategyDeepContext {
		return strategy
	}

	// 2. Dynamic UCB1 for strategy: "auto"
	if b.totalPulls < 10 {
		// Warm-up phase: pick Tier 1 for short text, Tier 2 for normal text
		if len(req.Text) < 25 {
			return engine.StrategyUltraFast
		}
		return engine.StrategyBalanced
	}

	bestArm := "tier2"
	bestScore := -math.MaxFloat64

	for key, arm := range b.arms {
		var ucbScore float64
		if arm.PullCount == 0 {
			ucbScore = 100.0 // Prioritize unpulled arms
		} else {
			// UCB1 formula: Q(a) + c * sqrt(ln(N) / N_a)
			exploration := b.cExploration * math.Sqrt(math.Log(float64(b.totalPulls))/float64(arm.PullCount))
			// Latency penalty adjustment
			latencyPenalty := (arm.AvgLatencyMs / 100.0) * 0.1
			ucbScore = arm.AverageReward + exploration - latencyPenalty
		}

		if ucbScore > bestScore {
			bestScore = ucbScore
			bestArm = key
		}
	}

	switch bestArm {
	case "tier1":
		return engine.StrategyUltraFast
	case "tier2":
		return engine.StrategyBalanced
	case "tier3":
		return engine.StrategyDeepContext
	default:
		return engine.StrategyBalanced
	}
}

// RecordReward updates empirical arm statistics and appends to the experience replay log.
func (b *ContextualBandit) RecordReward(fb *engine.FeedbackRequest) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.totalPulls++

	// Find corresponding arm
	var targetArm *ArmStats
	for _, arm := range b.arms {
		if arm.Name == fb.EngineUsed || fmt.Sprintf("tier%d", arm.Tier) == fb.EngineUsed {
			targetArm = arm
			break
		}
	}
	if targetArm == nil {
		targetArm = b.arms["tier2"] // Default
	}

	// Online incremental mean update
	targetArm.PullCount++
	targetArm.TotalReward += fb.Reward
	targetArm.AverageReward = targetArm.TotalReward / float64(targetArm.PullCount)

	if fb.LatencyMs > 0 {
		targetArm.AvgLatencyMs = (targetArm.AvgLatencyMs*0.9) + (fb.LatencyMs*0.1)
	}

	// Append experience to JSONL log
	record := ExperienceRecord{
		Timestamp:      time.Now().UTC().Format(time.RFC3339),
		RequestID:      fb.RequestID,
		Text:           fb.Text,
		EngineUsed:     fb.EngineUsed,
		PredictedLabel: fb.PredictedLabel,
		CorrectLabel:   fb.CorrectLabel,
		Reward:         fb.Reward,
		LatencyMs:      fb.LatencyMs,
		Comment:        fb.Comment,
	}

	return b.appendLog(record)
}

func (b *ContextualBandit) appendLog(rec ExperienceRecord) error {
	if b.logPath == "" {
		return nil
	}
	dir := filepath.Dir(b.logPath)
	_ = os.MkdirAll(dir, 0755)

	f, err := os.OpenFile(b.logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	data, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	_, err = f.Write(append(data, '\n'))
	return err
}

// Stats returns a snapshot of bandit parameters and arm metrics.
func (b *ContextualBandit) Stats() map[string]interface{} {
	b.mu.RLock()
	defer b.mu.RUnlock()

	armsMap := make(map[string]interface{})
	for k, v := range b.arms {
		armsMap[k] = map[string]interface{}{
			"engine":         v.Name,
			"tier":           v.Tier,
			"pull_count":     v.PullCount,
			"average_reward": math.Round(v.AverageReward*1000) / 1000,
			"avg_latency_ms": math.Round(v.AvgLatencyMs*100) / 100,
		}
	}

	return map[string]interface{}{
		"total_feedback_pulls": b.totalPulls,
		"exploration_factor":   b.cExploration,
		"arms":                 armsMap,
	}
}
