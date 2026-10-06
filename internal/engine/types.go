package engine

import (
	"context"
)

// TaskType represents the analysis granularity or objective.
type TaskType string

const (
	TaskDocumentLevel TaskType = "document_level"
	TaskAspectBased   TaskType = "aspect_based"
	TaskMultilingual  TaskType = "multilingual"
	TaskFineGrained   TaskType = "fine_grained"
)

// Strategy specifies the routing SLA and engine preference.
type Strategy string

const (
	StrategyUltraFast   Strategy = "ultra_fast"   // Tier 1 (< 1ms)
	StrategyBalanced    Strategy = "balanced"     // Tier 2 (15-40ms)
	StrategyDeepContext Strategy = "deep_context" // Tier 3 (LLM Fallback)
)

// Request defines the incoming sentiment analysis payload.
type Request struct {
	Text     string   `json:"text"`
	Task     string   `json:"task"`
	Strategy string   `json:"strategy,omitempty"`
	Language string   `json:"language,omitempty"`
	Aspects  []string `json:"aspects,omitempty"`
}

// Response defines the standardized response payload matching the Sentix API specification.
type Response struct {
	Status     string          `json:"status"`
	Task       string          `json:"task"`
	EngineUsed string          `json:"engine_used"`
	LatencyMs  float64         `json:"latency_ms"`
	Overall    *SentimentScore `json:"overall,omitempty"`
	Aspects    []AspectResult  `json:"aspects,omitempty"`
}

// SentimentScore represents the predicted sentiment label and confidence score.
type SentimentScore struct {
	Label string  `json:"label"`
	Score float64 `json:"score"`
}

// AspectResult holds aspect-level sentiment classification for ABSA tasks.
type AspectResult struct {
	Aspect     string  `json:"aspect"`
	Sentiment  string  `json:"sentiment"`
	Confidence float64 `json:"confidence"`
}

// Analyzer is the pluggable contract implemented by all model tiers.
type Analyzer interface {
	Name() string
	Tier() int
	Analyze(ctx context.Context, req *Request) (*Response, error)
	Close() error
}
