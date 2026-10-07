package tier2

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/ervijay/polysent/internal/engine"
	"github.com/ervijay/polysent/internal/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	ortEnvInitOnce sync.Once
	ortEnvErr      error
)

// Config defines the configuration for the Tier-2 ONNX inference engine.
type Config struct {
	ModelPath     string
	TokenizerPath string
	MetadataPath  string
	SharedLibPath string
	PoolSize      int
}

// ONNXEngine implements engine.Analyzer for Tier-2 high-accuracy Transformer inference.
type ONNXEngine struct {
	cfg         Config
	tokenizer   *tokenizer.WordPieceTokenizer
	sessionPool chan *ort.DynamicAdvancedSession
	id2label    map[int]string
	numClasses  int
	mu          sync.Mutex
	isClosed    bool
}

// NewONNXEngine initializes the ONNX Runtime environment, loads model metadata & tokenizer,
// and warms up a worker pool of dynamic inference sessions.
func NewONNXEngine(cfg Config) (*ONNXEngine, error) {
	if cfg.PoolSize <= 0 {
		cfg.PoolSize = 4
	}

	// 1. Initialize ONNX Runtime shared library & environment once
	ortEnvInitOnce.Do(func() {
		if !ort.IsInitialized() {
			if cfg.SharedLibPath != "" {
				ort.SetSharedLibraryPath(cfg.SharedLibPath)
			}
			ortEnvErr = ort.InitializeEnvironment()
		}
	})
	if ortEnvErr != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime environment: %w", ortEnvErr)
	}

	// 2. Load Tokenizer
	tok, err := tokenizer.NewWordPieceTokenizer(cfg.TokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer from %s: %w", cfg.TokenizerPath, err)
	}

	// 3. Load Metadata & Class Labels
	id2label := map[int]string{0: "NEGATIVE", 1: "POSITIVE"}
	numClasses := 2

	if cfg.MetadataPath != "" {
		if metaBytes, err := os.ReadFile(cfg.MetadataPath); err == nil {
			var meta struct {
				NumClasses int                 `json:"num_classes"`
				ID2Label   map[string]string   `json:"id2label"`
			}
			if err := json.Unmarshal(metaBytes, &meta); err == nil {
				if meta.NumClasses > 0 {
					numClasses = meta.NumClasses
				}
				if len(meta.ID2Label) > 0 {
					id2label = make(map[int]string, len(meta.ID2Label))
					for k, v := range meta.ID2Label {
						var id int
						fmt.Sscanf(k, "%d", &id)
						id2label[id] = v
					}
				}
			}
		}
	}

	// 4. Initialize Session Pool
	sessionPool := make(chan *ort.DynamicAdvancedSession, cfg.PoolSize)
	for i := 0; i < cfg.PoolSize; i++ {
		session, err := ort.NewDynamicAdvancedSession(
			cfg.ModelPath,
			[]string{"input_ids", "attention_mask"},
			[]string{"logits"},
			nil,
		)
		if err != nil {
			// Clean up already initialized sessions
			close(sessionPool)
			for s := range sessionPool {
				s.Destroy()
			}
			return nil, fmt.Errorf("failed to create session %d: %w", i, err)
		}
		sessionPool <- session
	}

	return &ONNXEngine{
		cfg:         cfg,
		tokenizer:   tok,
		sessionPool: sessionPool,
		id2label:    id2label,
		numClasses:  numClasses,
	}, nil
}

func (e *ONNXEngine) Name() string {
	return "onnx_distilbert_int8"
}

func (e *ONNXEngine) Tier() int {
	return 2
}

func (e *ONNXEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.isClosed {
		return nil
	}
	e.isClosed = true

	close(e.sessionPool)
	for session := range e.sessionPool {
		session.Destroy()
	}
	return nil
}

func (e *ONNXEngine) Analyze(ctx context.Context, req *engine.Request) (*engine.Response, error) {
	start := time.Now()

	// 1. Tokenize input text (max sequence length 128)
	encoded := e.tokenizer.Encode(req.Text, 128)
	seqLen := int64(len(encoded.InputIDs))

	// 2. Prepare ONNX input tensors
	shape := ort.NewShape(1, seqLen)
	inputIDsTensor, err := ort.NewTensor(shape, encoded.InputIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate input_ids tensor: %w", err)
	}
	defer inputIDsTensor.Destroy()

	attentionMaskTensor, err := ort.NewTensor(shape, encoded.AttentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	outputShape := ort.NewShape(1, int64(e.numClasses))
	logitsTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate logits tensor: %w", err)
	}
	defer logitsTensor.Destroy()

	// 3. Acquire a session from the worker pool
	var session *ort.DynamicAdvancedSession
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case s, ok := <-e.sessionPool:
		if !ok {
			return nil, fmt.Errorf("engine session pool is closed")
		}
		session = s
	}
	defer func() {
		// Return session back to pool safely without panicking on closed channel
		e.mu.Lock()
		defer e.mu.Unlock()
		if e.isClosed {
			session.Destroy()
		} else {
			e.sessionPool <- session
		}
	}()

	// 4. Run inference
	err = session.Run(
		[]ort.Value{inputIDsTensor, attentionMaskTensor},
		[]ort.Value{logitsTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("onnx inference execution error: %w", err)
	}

	// 5. Compute Softmax over logits
	rawLogits := logitsTensor.GetData()
	maxLogit := float64(rawLogits[0])
	for _, l := range rawLogits[1:] {
		if float64(l) > maxLogit {
			maxLogit = float64(l)
		}
	}

	expSum := 0.0
	expVals := make([]float64, len(rawLogits))
	for i, l := range rawLogits {
		ev := math.Exp(float64(l) - maxLogit)
		expVals[i] = ev
		expSum += ev
	}

	bestIdx := 0
	bestProb := 0.0
	for i, ev := range expVals {
		prob := ev / expSum
		if prob > bestProb {
			bestProb = prob
			bestIdx = i
		}
	}

	elapsed := float64(time.Since(start).Microseconds()) / 1000.0

	bestLabel := strings.ToLower(e.id2label[bestIdx])
	if bestLabel == "" {
		bestLabel = fmt.Sprintf("class_%d", bestIdx)
	}

	// Format response
	resp := &engine.Response{
		Status:     "success",
		Task:       req.Task,
		EngineUsed: e.Name(),
		LatencyMs:  elapsed,
		Overall: &engine.SentimentScore{
			Label: bestLabel,
			Score: bestProb,
		},
	}

	// Handle aspect_based task if aspects were provided
	if req.Task == string(engine.TaskAspectBased) && len(req.Aspects) > 0 {
		aspectResults := make([]engine.AspectResult, 0, len(req.Aspects))
		for _, aspect := range req.Aspects {
			// For sequence-level classification, each aspect inherits context-conditioned polarity
			aspectResults = append(aspectResults, engine.AspectResult{
				Aspect:     aspect,
				Sentiment:  bestLabel,
				Confidence: bestProb,
			})
		}
		resp.Aspects = aspectResults
	}

	return resp, nil
}
