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
	// ErrInvalidTokenizerJSON reports an invalid supported tokenizer.json asset.
	ErrInvalidTokenizerJSON = errors.New("tokenizer: invalid tokenizer.json")
	// ErrUnsupportedTokenizerJSON reports an unsupported tokenizer composition.
	ErrUnsupportedTokenizerJSON = errors.New("tokenizer: unsupported tokenizer.json")
)

// ByteLevelTokenizer executes a supported Hugging Face ByteLevel-BPE asset.
type ByteLevelTokenizer struct {
	model       *ByteLevelBPE
	prefixSpace bool
	added       map[string]int
	byID        map[int]addedToken
	ordered     []string
}

type addedToken struct {
	content string
	special bool
}

// SpecialTokenIDs identifies configured control tokens. A missing token is -1.
type SpecialTokenIDs struct {
	BOS int
	EOS int
	PAD int
	UNK int
}

// LoadByteLevelTokenizerAssets loads tokenizer.json and optional tokenizer_config.json
// from dir. Missing optional special-token fields remain -1.
func LoadByteLevelTokenizerAssets(dir string) (*ByteLevelTokenizer, SpecialTokenIDs, error) {
	tokenizer, err := LoadByteLevelTokenizer(dir + string(os.PathSeparator) + "tokenizer.json")
	if err != nil {
		return nil, SpecialTokenIDs{}, err
	}
	ids := SpecialTokenIDs{BOS: -1, EOS: -1, PAD: -1, UNK: -1}
	data, err := os.ReadFile(dir + string(os.PathSeparator) + "tokenizer_config.json")
	if errors.Is(err, os.ErrNotExist) {
		return tokenizer, ids, nil
	}
	if err != nil {
		return nil, SpecialTokenIDs{}, err
	}
	var config map[string]json.RawMessage
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, SpecialTokenIDs{}, fmt.Errorf("%w: tokenizer_config.json", ErrInvalidTokenizerJSON)
	}
	for _, field := range []struct {
		name string
		dest *int
	}{
		{"bos_token", &ids.BOS}, {"eos_token", &ids.EOS}, {"pad_token", &ids.PAD}, {"unk_token", &ids.UNK},
	} {
		raw, exists := config[field.name]
		if !exists || string(raw) == "null" {
			continue
		}
		token, err := tokenContent(raw)
		if err != nil {
			return nil, SpecialTokenIDs{}, fmt.Errorf("%w: %s", ErrInvalidTokenizerJSON, field.name)
		}
		id, exists := tokenizer.TokenID(token)
		if !exists {
			return nil, SpecialTokenIDs{}, fmt.Errorf("%w: %s %q", ErrInvalidTokenizerJSON, field.name, token)
		}
		*field.dest = id
	}
	return tokenizer, ids, nil
}

// LoadByteLevelTokenizer loads a BPE tokenizer.json with ByteLevel pre-tokenizer
// and decoder. Other tokenizer compositions return ErrUnsupportedTokenizerJSON.
func LoadByteLevelTokenizer(path string) (*ByteLevelTokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var document struct {
		PreTokenizer  json.RawMessage `json:"pre_tokenizer"`
		Decoder       json.RawMessage `json:"decoder"`
		Normalizer    json.RawMessage `json:"normalizer"`
		PostProcessor json.RawMessage `json:"post_processor"`
		AddedTokens   []struct {
			ID         int    `json:"id"`
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			SingleWord bool   `json:"single_word"`
			LStrip     bool   `json:"lstrip"`
			RStrip     bool   `json:"rstrip"`
			Normalized bool   `json:"normalized"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidTokenizerJSON, err)
	}
	prefixSpace, err := byteLevelConfig(document.PreTokenizer)
	if err != nil {
		return nil, err
	}
	if _, err := byteLevelConfig(document.Decoder); err != nil {
		return nil, err
	}
	if !supportedNormalizer(document.Normalizer) || !supportedPostProcessor(document.PostProcessor) {
		return nil, ErrUnsupportedTokenizerJSON
	}
	bpe, err := LoadBPE(path)
	if err != nil {
		return nil, err
	}
	model, err := NewByteLevelBPE(bpe)
	if err != nil {
		return nil, err
	}
	tokenizer := &ByteLevelTokenizer{
		model: model, prefixSpace: prefixSpace,
		added: make(map[string]int, len(document.AddedTokens)), byID: make(map[int]addedToken, len(document.AddedTokens)),
	}
	for _, token := range document.AddedTokens {
		if token.ID < 0 || token.Content == "" || token.SingleWord || token.LStrip || token.RStrip || token.Normalized {
			return nil, ErrInvalidTokenizerJSON
		}
		if _, exists := tokenizer.added[token.Content]; exists {
			return nil, fmt.Errorf("%w: duplicate added token %q", ErrInvalidTokenizerJSON, token.Content)
		}
		if _, exists := tokenizer.byID[token.ID]; exists {
			return nil, fmt.Errorf("%w: duplicate added token ID %d", ErrInvalidTokenizerJSON, token.ID)
		}
		tokenizer.added[token.Content] = token.ID
		tokenizer.byID[token.ID] = addedToken{content: token.Content, special: token.Special}
		tokenizer.ordered = append(tokenizer.ordered, token.Content)
	}
	sort.Slice(tokenizer.ordered, func(i, j int) bool {
		return len(tokenizer.ordered[i]) > len(tokenizer.ordered[j])
	})
	return tokenizer, nil
}

func isNull(raw json.RawMessage) bool {
	return len(raw) == 0 || strings.TrimSpace(string(raw)) == "null"
}

func byteLevelConfig(raw json.RawMessage) (bool, error) {
	var config struct {
		Type           string            `json:"type"`
		AddPrefixSpace bool              `json:"add_prefix_space"`
		PreTokenizers  []json.RawMessage `json:"pretokenizers"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &config) != nil {
		return false, ErrUnsupportedTokenizerJSON
	}
	if config.Type == "ByteLevel" {
		return config.AddPrefixSpace, nil
	}
	if config.Type == "Sequence" {
		for _, child := range config.PreTokenizers {
			prefixSpace, err := byteLevelConfig(child)
			if err == nil {
				return prefixSpace, nil
			}
		}
	}
	return false, ErrUnsupportedTokenizerJSON
}

func supportedNormalizer(raw json.RawMessage) bool {
	if isNull(raw) {
		return true
	}
	var config struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &config) != nil {
		return false
	}
	return config.Type == "NFC"
}

func supportedPostProcessor(raw json.RawMessage) bool {
	if isNull(raw) {
		return true
	}
	var config struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &config) != nil {
		return false
	}
	return config.Type == "ByteLevel"
}

// Encode converts text to IDs, matching added tokens before ByteLevel BPE.
func (t *ByteLevelTokenizer) Encode(text string) ([]int, error) {
	if t == nil || t.model == nil {
		return nil, ErrInvalidTokenizerJSON
	}
	ids := make([]int, 0, len(text))
	start, atStart := 0, true
	for start < len(text) {
		index, token := t.nextAdded(text[start:])
		if index < 0 {
			encoded, err := (ByteLevel{AddPrefixSpace: t.prefixSpace && atStart}).Encode(t.model, text[start:])
			if err != nil {
				return nil, err
			}
			return append(ids, encoded...), nil
		}
		if index > 0 {
			encoded, err := (ByteLevel{AddPrefixSpace: t.prefixSpace && atStart}).Encode(t.model, text[start:start+index])
			if err != nil {
				return nil, err
			}
			ids = append(ids, encoded...)
		}
		ids = append(ids, t.added[token])
		start += index + len(token)
		atStart = false
	}
	return ids, nil
}

// Decode converts IDs to text. skipSpecialTokens omits added special tokens.
func (t *ByteLevelTokenizer) Decode(ids []int, skipSpecialTokens bool) (string, error) {
	if t == nil || t.model == nil {
		return "", ErrInvalidTokenizerJSON
	}
	var result strings.Builder
	plain := make([]int, 0, len(ids))
	flush := func() error {
		if len(plain) == 0 {
			return nil
		}
		decoded, err := t.model.Decode(plain)
		if err != nil {
			return err
		}
		result.WriteString(decoded)
		plain = plain[:0]
		return nil
	}
	for _, id := range ids {
		added, exists := t.byID[id]
		if !exists {
			plain = append(plain, id)
			continue
		}
		if err := flush(); err != nil {
			return "", err
		}
		if !skipSpecialTokens || !added.special {
			result.WriteString(added.content)
		}
	}
	if err := flush(); err != nil {
		return "", err
	}
	return result.String(), nil
}

// AddedTokenID returns an added-token ID and whether it is known.
func (t *ByteLevelTokenizer) AddedTokenID(token string) (int, bool) {
	id, exists := t.added[token]
	return id, exists
}

// TokenID returns an added-token or BPE-vocabulary ID.
func (t *ByteLevelTokenizer) TokenID(token string) (int, bool) {
	if id, exists := t.added[token]; exists {
		return id, true
	}
	id, exists := t.model.bpe.vocab[token]
	return id, exists
}

func tokenContent(raw json.RawMessage) (string, error) {
	var content string
	if err := json.Unmarshal(raw, &content); err == nil && content != "" {
		return content, nil
	}
	var token struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(raw, &token); err != nil || token.Content == "" {
		return "", ErrInvalidTokenizerJSON
	}
	return token.Content, nil
}

func (t *ByteLevelTokenizer) nextAdded(text string) (int, string) {
	index, matched := -1, ""
	for _, token := range t.ordered {
		at := strings.Index(text, token)
		if at >= 0 && (index < 0 || at < index || at == index && len(token) > len(matched)) {
			index, matched = at, token
		}
	}
	return index, matched
}
