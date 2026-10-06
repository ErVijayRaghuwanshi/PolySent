package router

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/ervijay/polysent/internal/engine"
)

// Metrics records operational routing statistics across performance tiers.
type Metrics struct {
	TotalRequests int64 `json:"total_requests"`
	Tier1Hits     int64 `json:"tier1_hits"`
	Tier2Hits     int64 `json:"tier2_hits"`
	Tier3Hits     int64 `json:"tier3_hits"`
	Fallbacks     int64 `json:"fallbacks"`
}

// Router orchestrates SLA-driven dynamic routing across inference tiers.
type Router struct {
	tier1   engine.Analyzer
	tier2   engine.Analyzer
	tier3   engine.Analyzer
	metrics Metrics
}

// NewRouter constructs a router with registered analyzers.
func NewRouter(tier1, tier2, tier3 engine.Analyzer) *Router {
	return &Router{
		tier1: tier1,
		tier2: tier2,
		tier3: tier3,
	}
}

// GetMetrics returns snapshot of routing telemetry.
func (r *Router) GetMetrics() Metrics {
	return Metrics{
		TotalRequests: atomic.LoadInt64(&r.metrics.TotalRequests),
		Tier1Hits:     atomic.LoadInt64(&r.metrics.Tier1Hits),
		Tier2Hits:     atomic.LoadInt64(&r.metrics.Tier2Hits),
		Tier3Hits:     atomic.LoadInt64(&r.metrics.Tier3Hits),
		Fallbacks:     atomic.LoadInt64(&r.metrics.Fallbacks),
	}
}

// Route directs incoming requests to the optimal engine based on SLA strategy and availability.
func (r *Router) Route(ctx context.Context, req *engine.Request) (*engine.Response, error) {
	atomic.AddInt64(&r.metrics.TotalRequests, 1)

	strategy := engine.Strategy(req.Strategy)
	if strategy == "" || strategy == "auto" {
		// Dynamic decision: short straightforward text uses Tier 1; otherwise Tier 2
		if len(req.Text) < 25 && r.tier1 != nil {
			strategy = engine.StrategyUltraFast
		} else if r.tier2 != nil {
			strategy = engine.StrategyBalanced
		} else {
			strategy = engine.StrategyUltraFast
		}
	}

	switch strategy {
	case engine.StrategyUltraFast:
		if r.tier1 != nil {
			atomic.AddInt64(&r.metrics.Tier1Hits, 1)
			return r.tier1.Analyze(ctx, req)
		}

	case engine.StrategyBalanced:
		if r.tier2 != nil {
			resp, err := r.tier2.Analyze(ctx, req)
			if err == nil {
				atomic.AddInt64(&r.metrics.Tier2Hits, 1)
				return resp, nil
			}
			// Fallback to Tier 1 on Tier 2 failure
			atomic.AddInt64(&r.metrics.Fallbacks, 1)
		}
		if r.tier1 != nil {
			atomic.AddInt64(&r.metrics.Tier1Hits, 1)
			return r.tier1.Analyze(ctx, req)
		}

	case engine.StrategyDeepContext:
		if r.tier3 != nil {
			resp, err := r.tier3.Analyze(ctx, req)
			if err == nil {
				atomic.AddInt64(&r.metrics.Tier3Hits, 1)
				return resp, nil
			}
			atomic.AddInt64(&r.metrics.Fallbacks, 1)
		}
		// Fallback to Tier 2 or Tier 1
		if r.tier2 != nil {
			atomic.AddInt64(&r.metrics.Tier2Hits, 1)
			return r.tier2.Analyze(ctx, req)
		}
		if r.tier1 != nil {
			atomic.AddInt64(&r.metrics.Tier1Hits, 1)
			return r.tier1.Analyze(ctx, req)
		}
	}

	// Ultimate fallback to whichever analyzer is non-nil
	if r.tier1 != nil {
		atomic.AddInt64(&r.metrics.Tier1Hits, 1)
		return r.tier1.Analyze(ctx, req)
	}

	return nil, fmt.Errorf("no inference engine available to process request")
}

// AvailableEngines returns status map of loaded analyzers.
func (r *Router) AvailableEngines() map[string]string {
	status := make(map[string]string)
	if r.tier1 != nil {
		status["tier1_tfidf"] = "ready"
	} else {
		status["tier1_tfidf"] = "unconfigured"
	}

	if r.tier2 != nil {
		status["tier2_onnx"] = "ready"
	} else {
		status["tier2_onnx"] = "unconfigured"
	}

	if r.tier3 != nil {
		status["tier3_llm"] = "ready"
	} else {
		status["tier3_llm"] = "unconfigured"
	}
	return status
}
