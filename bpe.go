package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
)

var (
	// ErrInvalidBPE reports invalid BPE vocabulary or merge data.
	ErrInvalidBPE = errors.New("tokenizer: invalid BPE model")
	// ErrUnknownToken reports text that cannot be represented by a BPE model.
	ErrUnknownToken = errors.New("tokenizer: unknown token")
	// ErrUnsupportedModel reports a tokenizer model not implemented by this package.
	ErrUnsupportedModel = errors.New("tokenizer: unsupported tokenizer model")
)

// Encoder encodes pre-tokenized text and decodes token IDs.
// Callers select a normalizer and pre-tokenizer separately.
type Encoder interface {
	EncodePiece(string) ([]int, error)
	Decode([]int) (string, error)
}

// BPE is a deterministic byte-agnostic BPE merge model.
// It operates on one pre-tokenized piece at a time.
type BPE struct {
	vocab   map[string]int
	reverse map[int]string
	merges  map[string]int
	unknown int
	hasUNK  bool
}

// NewBPE validates and creates a BPE model. Each merge is an ordered token pair.
// unknownToken may be empty when unknown input must return ErrUnknownToken.
func NewBPE(vocabulary map[string]int, merges [][2]string, unknownToken string) (*BPE, error) {
	if len(vocabulary) == 0 {
		return nil, ErrInvalidBPE
	}
	model := &BPE{
		vocab: make(map[string]int, len(vocabulary)), reverse: make(map[int]string, len(vocabulary)),
		merges: make(map[string]int, len(merges)),
	}
	for token, id := range vocabulary {
		if token == "" || id < 0 {
			return nil, ErrInvalidBPE
		}
		if _, exists := model.reverse[id]; exists {
			return nil, fmt.Errorf("%w: duplicate ID %d", ErrInvalidBPE, id)
		}
		model.vocab[token], model.reverse[id] = id, token
	}
	if unknownToken != "" {
		id, exists := model.vocab[unknownToken]
		if !exists {
			return nil, fmt.Errorf("%w: unknown token %q", ErrInvalidBPE, unknownToken)
		}
		model.unknown, model.hasUNK = id, true
	}
	for rank, merge := range merges {
		if merge[0] == "" || merge[1] == "" {
			return nil, ErrInvalidBPE
		}
		key := pairKey(merge[0], merge[1])
		if _, exists := model.merges[key]; exists {
			return nil, fmt.Errorf("%w: duplicate merge %q", ErrInvalidBPE, key)
		}
		model.merges[key] = rank
	}
	return model, nil
}

// EncodePiece merges one pre-tokenized text piece into BPE IDs.
func (b *BPE) EncodePiece(piece string) ([]int, error) {
	if b == nil || piece == "" {
		return nil, ErrInvalidBPE
	}
	tokens := make([]string, 0, len(piece))
	for _, value := range piece {
		tokens = append(tokens, string(value))
	}
	for {
		index, found := b.nextMerge(tokens)
		if !found {
			break
		}
		tokens[index] += tokens[index+1]
		copy(tokens[index+1:], tokens[index+2:])
		tokens = tokens[:len(tokens)-1]
	}
	ids := make([]int, len(tokens))
	for index, token := range tokens {
		id, exists := b.vocab[token]
		if !exists {
			if !b.hasUNK {
				return nil, fmt.Errorf("%w: %q", ErrUnknownToken, token)
			}
			id = b.unknown
		}
		ids[index] = id
	}
	return ids, nil
}

// Decode joins the tokens represented by ids without applying a byte decoder.
func (b *BPE) Decode(ids []int) (string, error) {
	if b == nil {
		return "", ErrInvalidBPE
	}
	var result strings.Builder
	for _, id := range ids {
		token, exists := b.reverse[id]
		if !exists {
			return "", fmt.Errorf("%w: ID %d", ErrUnknownToken, id)
		}
		result.WriteString(token)
	}
	return result.String(), nil
}

func (b *BPE) nextMerge(tokens []string) (int, bool) {
	bestIndex, bestRank := 0, 0
	found := false
	for index := 0; index+1 < len(tokens); index++ {
		rank, exists := b.merges[pairKey(tokens[index], tokens[index+1])]
		if exists && (!found || rank < bestRank) {
			bestIndex, bestRank, found = index, rank, true
		}
	}
	return bestIndex, found
}

func pairKey(left, right string) string { return left + "\x00" + right }

// LoadBPE reads a Hugging Face tokenizer.json file whose model type is BPE.
// Pre-tokenizer, normalizer, and byte-decoder execution are intentionally not
// inferred from this function.
func LoadBPE(path string) (*BPE, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var document struct {
		Model json.RawMessage
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, err
	}
	var model struct {
		Type   string
		Vocab  map[string]int
		Merges []json.RawMessage
	}
	if err := json.Unmarshal(document.Model, &model); err != nil {
		return nil, err
	}
	if !strings.EqualFold(model.Type, "BPE") {
		return nil, fmt.Errorf("%w: %s", ErrUnsupportedModel, model.Type)
	}
	merges := make([][2]string, len(model.Merges))
	for index, raw := range model.Merges {
		merge, err := parseMerge(raw)
		if err != nil {
			return nil, err
		}
		merges[index] = merge
	}
	var rawModel map[string]json.RawMessage
	if err := json.Unmarshal(document.Model, &rawModel); err != nil {
		return nil, err
	}
	var unknown string
	if raw, exists := rawModel["unk_token"]; exists {
		if err := json.Unmarshal(raw, &unknown); err != nil {
			return nil, ErrInvalidBPE
		}
	}
	return NewBPE(model.Vocab, merges, unknown)
}

func parseMerge(raw json.RawMessage) ([2]string, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		parts := strings.SplitN(text, " ", 2)
		if len(parts) == 2 && parts[0] != "" && parts[1] != "" {
			return [2]string{parts[0], parts[1]}, nil
		}
	}
	var parts []string
	if err := json.Unmarshal(raw, &parts); err == nil && len(parts) == 2 && parts[0] != "" && parts[1] != "" {
		return [2]string{parts[0], parts[1]}, nil
	}
	return [2]string{}, ErrInvalidBPE
}

// Vocabulary returns a copy intended for diagnostics and fixture validation.
func (b *BPE) Vocabulary() map[string]int {
	result := make(map[string]int, len(b.vocab))
	for token, id := range b.vocab {
		result[token] = id
	}
	return result
}

// IDs returns vocabulary IDs in ascending order for deterministic diagnostics.
func (b *BPE) IDs() []int {
	result := make([]int, 0, len(b.reverse))
	for id := range b.reverse {
		result = append(result, id)
	}
	sort.Ints(result)
	return result
}
