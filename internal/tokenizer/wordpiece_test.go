package tokenizer_test

import (
	"path/filepath"
	"testing"

	"github.com/ervijay/polysent/internal/tokenizer"
)

func TestWordPieceTokenizer(t *testing.T) {
	vocabPath := filepath.Join("..", "..", "models", "tier2_distilbert", "tokenizer.json")
	tok, err := tokenizer.NewWordPieceTokenizer(vocabPath)
	if err != nil {
		// Fallback to vocab.txt
		vocabPath = filepath.Join("..", "..", "models", "tier2_distilbert", "vocab.txt")
		tok, err = tokenizer.NewWordPieceTokenizer(vocabPath)
		if err != nil {
			t.Fatalf("Failed to initialize WordPieceTokenizer: %v", err)
		}
	}

	text := "The display is crisp and vivid, but customer service was slow."
	enc := tok.Encode(text, 64)

	if len(enc.InputIDs) == 0 {
		t.Fatalf("Expected non-empty input IDs")
	}

	// Verify [CLS] and [SEP]
	if enc.InputIDs[0] != tokenizer.ClsTokenID {
		t.Errorf("Expected first token to be [CLS] (%d), got %d", tokenizer.ClsTokenID, enc.InputIDs[0])
	}
	lastIdx := len(enc.InputIDs) - 1
	if enc.InputIDs[lastIdx] != tokenizer.SepTokenID {
		t.Errorf("Expected last token to be [SEP] (%d), got %d", tokenizer.SepTokenID, enc.InputIDs[lastIdx])
	}

	// Verify attention mask length matches input IDs
	if len(enc.AttentionMask) != len(enc.InputIDs) {
		t.Errorf("Attention mask length (%d) mismatch with input IDs (%d)", len(enc.AttentionMask), len(enc.InputIDs))
	}

	for i, mask := range enc.AttentionMask {
		if mask != 1 {
			t.Errorf("Expected attention mask at index %d to be 1, got %d", i, mask)
		}
	}

	t.Logf("Tokenized '%s' into %d tokens: %v", text, len(enc.Tokens), enc.Tokens)
}

func TestWordPieceUnicodeHandling(t *testing.T) {
	vocabPath := filepath.Join("..", "..", "models", "tier2_distilbert", "tokenizer.json")
	tok, err := tokenizer.NewWordPieceTokenizer(vocabPath)
	if err != nil {
		vocabPath = filepath.Join("..", "..", "models", "tier2_distilbert", "vocab.txt")
		tok, err = tokenizer.NewWordPieceTokenizer(vocabPath)
		if err != nil {
			t.Fatalf("Failed to initialize WordPieceTokenizer: %v", err)
		}
	}

	// Test unicode accents, non-ASCII chars, and edge maxLen
	unicodeText := "Café naïve résumé with multilingual text हिंदी and emoji 👍!"
	enc := tok.Encode(unicodeText, 64)
	if len(enc.InputIDs) < 2 {
		t.Fatalf("Expected valid tokenization for unicode text")
	}

	// Test maxLen = 1 defensive bound
	enc1 := tok.Encode("hello world", 1)
	if len(enc1.InputIDs) != 1 || enc1.InputIDs[0] != tokenizer.ClsTokenID {
		t.Errorf("Expected single CLS token for maxLen=1, got %v", enc1.InputIDs)
	}
}

func BenchmarkWordPieceTokenizer(b *testing.B) {
	vocabPath := filepath.Join("..", "..", "models", "tier2_distilbert", "tokenizer.json")
	tok, err := tokenizer.NewWordPieceTokenizer(vocabPath)
	if err != nil {
		vocabPath = filepath.Join("..", "..", "models", "tier2_distilbert", "vocab.txt")
		tok, err = tokenizer.NewWordPieceTokenizer(vocabPath)
		if err != nil {
			b.Fatalf("Failed to initialize WordPieceTokenizer: %v", err)
		}
	}

	text := "The display is crisp and vivid, but customer service was frustratingly slow."
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = tok.Encode(text, 64)
	}
}
