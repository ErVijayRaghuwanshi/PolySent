package tier1

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/ervijay/polysent/internal/engine"
)

var (
	wordRegex = regexp.MustCompile(`\b\w\w+\b`)
)

type weightsFile struct {
	ModelName      string             `json:"model_name"`
	Version        string             `json:"version"`
	Classes        []string           `json:"classes"`
	VocabularySize int                `json:"vocabulary_size"`
	SublinearTF    bool               `json:"sublinear_tf"`
	Vocabulary     map[string]int     `json:"vocabulary"`
	IDF            []float64          `json:"idf"`
	Coefficients   [][]float64        `json:"coefficients"`
	Intercepts     []float64          `json:"intercepts"`
}

// TFIDFAnalyzer implements engine.Analyzer for Tier-1 sub-millisecond inference.
type TFIDFAnalyzer struct {
	weights weightsFile
}

// NewTFIDFAnalyzer loads the exported weights and vocabulary JSON.
func NewTFIDFAnalyzer(weightsPath string) (*TFIDFAnalyzer, error) {
	data, err := os.ReadFile(weightsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read tier-1 weights: %w", err)
	}

	var w weightsFile
	if err := json.Unmarshal(data, &w); err != nil {
		return nil, fmt.Errorf("failed to parse tier-1 weights JSON: %w", err)
	}

	return &TFIDFAnalyzer{weights: w}, nil
}

func (a *TFIDFAnalyzer) Name() string {
	return "tier1_purego_tfidf_logreg"
}

func (a *TFIDFAnalyzer) Tier() int {
	return 1
}

func (a *TFIDFAnalyzer) Close() error {
	return nil
}

func (a *TFIDFAnalyzer) Analyze(ctx context.Context, req *engine.Request) (*engine.Response, error) {
	start := time.Now()

	// 1. Tokenize into lowercase tokens
	lower := strings.ToLower(req.Text)
	matches := wordRegex.FindAllString(lower, -1)
	if len(matches) == 0 {
		return &engine.Response{
			Status:     "success",
			Task:       req.Task,
			EngineUsed: a.Name(),
			LatencyMs:  float64(time.Since(start).Microseconds()) / 1000.0,
			Overall: &engine.SentimentScore{
				Label: "neutral",
				Score: 0.33,
			},
		}, nil
	}

	// 2. Build unigrams & bigrams
	terms := make([]string, 0, len(matches)*2)
	for i := 0; i < len(matches); i++ {
		terms = append(terms, matches[i])
		if i < len(matches)-1 {
			terms = append(terms, matches[i]+" "+matches[i+1])
		}
	}

	// 3. Count term occurrences
	tfCounts := make(map[int]int)
	for _, term := range terms {
		if idx, ok := a.weights.Vocabulary[term]; ok {
			tfCounts[idx]++
		}
	}

	// 4. Compute TF-IDF values and calculate Euclidean norm (L2)
	normSq := 0.0
	tfidfVals := make(map[int]float64, len(tfCounts))
	for idx, count := range tfCounts {
		var tf float64
		if a.weights.SublinearTF {
			tf = 1.0 + math.Log(float64(count))
		} else {
			tf = float64(count)
		}
		val := tf * a.weights.IDF[idx]
		tfidfVals[idx] = val
		normSq += val * val
	}

	l2Norm := 1.0
	if normSq > 0 {
		l2Norm = math.Sqrt(normSq)
	}

	// 5. Linear model scoring for each class
	numClasses := len(a.weights.Classes)
	logits := make([]float64, numClasses)
	for c := 0; c < numClasses; c++ {
		score := a.weights.Intercepts[c]
		coefs := a.weights.Coefficients[c]
		for idx, val := range tfidfVals {
			normVal := val / l2Norm
			score += normVal * coefs[idx]
		}
		logits[c] = score
	}

	// 6. Softmax
	maxLogit := logits[0]
	for _, l := range logits[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}

	expSum := 0.0
	expVals := make([]float64, numClasses)
	for i, l := range logits {
		ev := math.Exp(l - maxLogit)
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

	return &engine.Response{
		Status:     "success",
		Task:       req.Task,
		EngineUsed: a.Name(),
		LatencyMs:  elapsed,
		Overall: &engine.SentimentScore{
			Label: a.weights.Classes[bestIdx],
			Score: bestProb,
		},
	}, nil
}
