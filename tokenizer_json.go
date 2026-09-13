package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"golang.org/x/text/unicode/norm"
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
	qwenSplit   bool
	nfc         bool
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
	prefixSpace, qwenSplit, err := preTokenizerConfig(document.PreTokenizer)
	if err != nil {
		return nil, err
	}
	if err := decoderConfig(document.Decoder); err != nil {
		return nil, err
	}
	nfc, err := normalizerConfig(document.Normalizer)
	if err != nil {
		return nil, err
	}
	if err := postProcessorConfig(document.PostProcessor); err != nil {
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
		model: model, prefixSpace: prefixSpace, qwenSplit: qwenSplit, nfc: nfc,
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

const qwenSplitPattern = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"

func preTokenizerConfig(raw json.RawMessage) (bool, bool, error) {
	var config struct {
		Type           string            `json:"type"`
		AddPrefixSpace bool              `json:"add_prefix_space"`
		PreTokenizers  []json.RawMessage `json:"pretokenizers"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &config) != nil {
		return false, false, ErrUnsupportedTokenizerJSON
	}
	if config.Type == "ByteLevel" {
		return config.AddPrefixSpace, false, nil
	}
	if config.Type != "Sequence" || len(config.PreTokenizers) != 2 || !qwenSplitConfig(config.PreTokenizers[0]) {
		return false, false, ErrUnsupportedTokenizerJSON
	}
	var byteLevel struct {
		Type           string `json:"type"`
		AddPrefixSpace bool   `json:"add_prefix_space"`
		UseRegex       bool   `json:"use_regex"`
	}
	if json.Unmarshal(config.PreTokenizers[1], &byteLevel) != nil || byteLevel.Type != "ByteLevel" || byteLevel.AddPrefixSpace || byteLevel.UseRegex {
		return false, false, ErrUnsupportedTokenizerJSON
	}
	return false, true, nil
}

func qwenSplitConfig(raw json.RawMessage) bool {
	var config struct {
		Type     string                 `json:"type"`
		Pattern  struct{ Regex string } `json:"pattern"`
		Behavior string                 `json:"behavior"`
		Invert   bool                   `json:"invert"`
	}
	return json.Unmarshal(raw, &config) == nil && config.Type == "Split" && config.Pattern.Regex == qwenSplitPattern && config.Behavior == "Isolated" && !config.Invert
}

func decoderConfig(raw json.RawMessage) error {
	var config struct {
		Type string `json:"type"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &config) != nil || config.Type != "ByteLevel" {
		return ErrUnsupportedTokenizerJSON
	}
	return nil
}

func normalizerConfig(raw json.RawMessage) (bool, error) {
	if isNull(raw) {
		return false, nil
	}
	var config struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &config) != nil {
		return false, ErrUnsupportedTokenizerJSON
	}
	if config.Type != "NFC" {
		return false, ErrUnsupportedTokenizerJSON
	}
	return true, nil
}

func postProcessorConfig(raw json.RawMessage) error {
	if isNull(raw) {
		return nil
	}
	var config struct {
		Type   string            `json:"type"`
		Single []json.RawMessage `json:"single"`
		Pair   []json.RawMessage `json:"pair"`
	}
	if json.Unmarshal(raw, &config) != nil {
		return ErrUnsupportedTokenizerJSON
	}
	if config.Type == "ByteLevel" {
		return nil
	}
	if config.Type != "TemplateProcessing" || len(config.Single) != 1 || !templateSequence(config.Single[0], "A", 0) || len(config.Pair) != 2 || !templateSequence(config.Pair[0], "A", 0) || !templateSequence(config.Pair[1], "B", 1) {
		return ErrUnsupportedTokenizerJSON
	}
	return nil
}

func templateSequence(raw json.RawMessage, id string, typeID int) bool {
	var value struct {
		Sequence struct {
			ID     string `json:"id"`
			TypeID int    `json:"type_id"`
		} `json:"Sequence"`
	}
	return json.Unmarshal(raw, &value) == nil && value.Sequence.ID == id && value.Sequence.TypeID == typeID
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
			encoded, err := t.encodePlain(text[start:], atStart)
			if err != nil {
				return nil, err
			}
			return append(ids, encoded...), nil
		}
		if index > 0 {
			encoded, err := t.encodePlain(text[start:start+index], atStart)
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

func (t *ByteLevelTokenizer) encodePlain(text string, atStart bool) ([]int, error) {
	if t.nfc {
		text = norm.NFC.String(text)
	}
	return (ByteLevel{AddPrefixSpace: t.prefixSpace && atStart, Qwen: t.qwenSplit}).Encode(t.model, text)
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
