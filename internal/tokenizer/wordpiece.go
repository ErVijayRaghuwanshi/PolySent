package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

const (
	PadTokenID = int64(0)
	UnkTokenID = int64(100)
	ClsTokenID = int64(101)
	SepTokenID = int64(102)
)

// WordPieceTokenizer implements subword tokenization compatible with DistilBERT and BERT.
type WordPieceTokenizer struct {
	vocab      map[string]int64
	invVocab   map[int64]string
	maxSubword int
}

// NewWordPieceTokenizer loads vocabulary from either a tokenizer.json or a vocab.txt file.
func NewWordPieceTokenizer(path string) (*WordPieceTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open vocab/tokenizer file: %w", err)
	}

	vocab := make(map[string]int64)
	invVocab := make(map[int64]string)

	if strings.HasSuffix(path, ".json") {
		var tokJSON struct {
			Model struct {
				Vocab map[string]int64 `json:"vocab"`
			} `json:"model"`
		}
		if err := json.Unmarshal(data, &tokJSON); err == nil && len(tokJSON.Model.Vocab) > 0 {
			vocab = tokJSON.Model.Vocab
			for k, v := range vocab {
				invVocab[v] = k
			}
			return &WordPieceTokenizer{
				vocab:      vocab,
				invVocab:   invVocab,
				maxSubword: 100,
			}, nil
		}
	}

	// Line-by-line vocab.txt format
	scanner := bufio.NewScanner(strings.NewReader(string(data)))
	var idx int64

	for scanner.Scan() {
		token := strings.TrimSpace(scanner.Text())
		if token != "" {
			vocab[token] = idx
			invVocab[idx] = token
			idx++
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading vocab file: %w", err)
	}

	return &WordPieceTokenizer{
		vocab:      vocab,
		invVocab:   invVocab,
		maxSubword: 100,
	}, nil
}

// TokenizedInput holds token IDs and attention masks ready for ONNX runtime tensors.
type TokenizedInput struct {
	InputIDs      []int64
	AttentionMask []int64
	Tokens        []string
}

// Encode converts raw text into input_ids and attention_mask with [CLS] and [SEP].
func (t *WordPieceTokenizer) Encode(text string, maxLen int) *TokenizedInput {
	words := t.basicTokenize(text)
	subwords := make([]string, 0, len(words)*2)

	for _, word := range words {
		// WordPiece subword breakdown
		if len(word) > t.maxSubword {
			subwords = append(subwords, "[UNK]")
			continue
		}

		isBad := false
		start := 0
		subTokens := make([]string, 0)

		for start < len(word) {
			end := len(word)
			curSubstr := ""
			for start < end {
				substr := word[start:end]
				if start > 0 {
					substr = "##" + substr
				}
				if _, ok := t.vocab[substr]; ok {
					curSubstr = substr
					break
				}
				end--
			}

			if curSubstr == "" {
				isBad = true
				break
			}

			subTokens = append(subTokens, curSubstr)
			start = end
		}

		if isBad {
			subwords = append(subwords, "[UNK]")
		} else {
			subwords = append(subwords, subTokens...)
		}
	}

	// Truncate to leave room for [CLS] and [SEP]
	if maxLen > 2 && len(subwords) > maxLen-2 {
		subwords = subwords[:maxLen-2]
	}

	// Build final sequence: [CLS] + subwords + [SEP]
	finalTokens := make([]string, 0, len(subwords)+2)
	finalTokens = append(finalTokens, "[CLS]")
	finalTokens = append(finalTokens, subwords...)
	finalTokens = append(finalTokens, "[SEP]")

	inputIDs := make([]int64, len(finalTokens))
	attentionMask := make([]int64, len(finalTokens))

	for i, tok := range finalTokens {
		if id, ok := t.vocab[tok]; ok {
			inputIDs[i] = id
		} else {
			inputIDs[i] = UnkTokenID
		}
		attentionMask[i] = 1
	}

	return &TokenizedInput{
		InputIDs:      inputIDs,
		AttentionMask: attentionMask,
		Tokens:        finalTokens,
	}
}

// basicTokenize performs lowercasing and punctuation separation.
func (t *WordPieceTokenizer) basicTokenize(text string) []string {
	lower := strings.ToLower(text)
	var words []string
	var cur strings.Builder

	for _, r := range lower {
		if unicode.IsSpace(r) {
			if cur.Len() > 0 {
				words = append(words, cur.String())
				cur.Reset()
			}
		} else if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			if cur.Len() > 0 {
				words = append(words, cur.String())
				cur.Reset()
			}
			words = append(words, string(r))
		} else {
			cur.WriteRune(r)
		}
	}

	if cur.Len() > 0 {
		words = append(words, cur.String())
	}

	return words
}
