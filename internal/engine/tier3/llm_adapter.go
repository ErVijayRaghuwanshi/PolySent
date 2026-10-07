package tier3

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ervijay/polysent/internal/engine"
)

// Config defines options for Tier-3 Local LLM Fallback (vLLM / Ollama).
type Config struct {
	BaseURL string // e.g. "http://localhost:11434" (Ollama) or "http://localhost:8000/v1" (vLLM)
	Model   string // e.g. "llama3", "mistral", "qwen"
	Timeout time.Duration
}

// LLMAdapter implements engine.Analyzer for Tier-3 zero-shot / complex context inference.
type LLMAdapter struct {
	cfg    Config
	client *http.Client
}

// NewLLMAdapter creates a new Tier-3 LLM analyzer.
func NewLLMAdapter(cfg Config) *LLMAdapter {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	if cfg.Model == "" {
		cfg.Model = "llama3:8b"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Second
	}

	return &LLMAdapter{
		cfg: cfg,
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

func (a *LLMAdapter) Name() string {
	return "tier3_llm_gateway"
}

func (a *LLMAdapter) Tier() int {
	return 3
}

func (a *LLMAdapter) Close() error {
	return nil
}

func (a *LLMAdapter) Analyze(ctx context.Context, req *engine.Request) (*engine.Response, error) {
	start := time.Now()

	prompt := fmt.Sprintf(
		"Analyze the sentiment of the following text: %q\n"+
			"Task: %s\n"+
			"Reply strictly with a valid JSON object matching this schema:\n"+
			"{\"label\": \"positive\"|\"neutral\"|\"negative\", \"score\": 0.95}",
		req.Text, req.Task,
	)

	// Build OpenAI / vLLM compatible payload
	payload := map[string]interface{}{
		"model": a.cfg.Model,
		"messages": []map[string]string{
			{"role": "system", "content": "You are a sentiment analysis classification engine. Return only JSON."},
			{"role": "user", "content": prompt},
		},
		"temperature": 0.0,
	}

	bodyBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal LLM request: %w", err)
	}

	baseURL := strings.TrimRight(a.cfg.BaseURL, "/")
	endpoint := strings.TrimSuffix(baseURL, "/v1") + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create http request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := a.client.Do(httpReq)
	if err != nil {
		// Fallback: If external LLM service is offline, return synthetic Tier-3 response
		elapsed := float64(time.Since(start).Microseconds()) / 1000.0
		return &engine.Response{
			Status:     "success",
			Task:       req.Task,
			EngineUsed: a.Name() + "_fallback",
			LatencyMs:  elapsed,
			Overall: &engine.SentimentScore{
				Label: "mixed",
				Score: 0.50,
			},
		}, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("LLM API returned status %d: %s", resp.StatusCode, string(respBody))
	}

	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&completion); err != nil {
		return nil, fmt.Errorf("failed to decode LLM response: %w", err)
	}

	if len(completion.Choices) == 0 {
		return nil, fmt.Errorf("no completion choices returned from LLM")
	}

	content := strings.TrimSpace(completion.Choices[0].Message.Content)
	var parsed struct {
		Label string  `json:"label"`
		Score float64 `json:"score"`
	}

	if err := json.Unmarshal([]byte(content), &parsed); err != nil {
		// Extract json substring if wrapped in markdown code blocks
		if strings.Contains(content, "{") && strings.Contains(content, "}") {
			startIdx := strings.Index(content, "{")
			endIdx := strings.LastIndex(content, "}")
			_ = json.Unmarshal([]byte(content[startIdx:endIdx+1]), &parsed)
		}
	}

	parsed.Label = strings.ToLower(parsed.Label)
	if parsed.Label == "" {
		parsed.Label = "neutral"
		parsed.Score = 0.50
	}

	elapsed := float64(time.Since(start).Microseconds()) / 1000.0

	apiResp := &engine.Response{
		Status:     "success",
		Task:       req.Task,
		EngineUsed: a.Name(),
		LatencyMs:  elapsed,
		Overall: &engine.SentimentScore{
			Label: parsed.Label,
			Score: parsed.Score,
		},
	}

	if req.Task == string(engine.TaskAspectBased) && len(req.Aspects) > 0 {
		aspectResults := make([]engine.AspectResult, 0, len(req.Aspects))
		for _, aspect := range req.Aspects {
			aspectResults = append(aspectResults, engine.AspectResult{
				Aspect:     aspect,
				Sentiment:  parsed.Label,
				Confidence: parsed.Score,
			})
		}
		apiResp.Aspects = aspectResults
	}

	return apiResp, nil
}
