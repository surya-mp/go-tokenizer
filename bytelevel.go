package tokenizer

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrInvalidByteLevel reports an invalid byte-level encoder or decoder.
	ErrInvalidByteLevel = errors.New("tokenizer: invalid byte-level encoding")
)

// ByteLevelBPE applies Hugging Face's GPT-2 byte-to-Unicode transform around a
// BPE model. It operates on a piece supplied by a separate pre-tokenizer.
type ByteLevelBPE struct {
	bpe     *BPE
	encoded [256]rune
	decoded map[rune]byte
}

// NewByteLevelBPE wraps model with the standard reversible byte encoding.
func NewByteLevelBPE(model *BPE) (*ByteLevelBPE, error) {
	if model == nil {
		return nil, ErrInvalidByteLevel
	}
	encoded := byteToRune()
	decoded := make(map[rune]byte, len(encoded))
	for value, token := range encoded {
		decoded[token] = byte(value)
	}
	return &ByteLevelBPE{bpe: model, encoded: encoded, decoded: decoded}, nil
}

// EncodePiece byte-encodes and merges one already pre-tokenized text piece.
func (b *ByteLevelBPE) EncodePiece(piece string) ([]int, error) {
	if b == nil || b.bpe == nil || piece == "" {
		return nil, ErrInvalidByteLevel
	}
	var encoded strings.Builder
	encoded.Grow(len(piece))
	for _, value := range []byte(piece) {
		encoded.WriteRune(b.encoded[value])
	}
	return b.bpe.EncodePiece(encoded.String())
}

// Decode reverses the byte encoding after BPE token decoding.
func (b *ByteLevelBPE) Decode(ids []int) (string, error) {
	if b == nil || b.bpe == nil {
		return "", ErrInvalidByteLevel
	}
	encoded, err := b.bpe.Decode(ids)
	if err != nil {
		return "", err
	}
	decoded := make([]byte, 0, len(encoded))
	for _, token := range encoded {
		value, exists := b.decoded[token]
		if !exists {
			return "", fmt.Errorf("%w: token %q", ErrInvalidByteLevel, token)
		}
		decoded = append(decoded, value)
	}
	return string(decoded), nil
}

func byteToRune() [256]rune {
	var result [256]rune
	used := make([]bool, 256)
	for _, span := range [][2]int{{'!', '~'}, {'¡', '¬'}, {'®', 'ÿ'}} {
		for value := span[0]; value <= span[1]; value++ {
			used[value] = true
			result[value] = rune(value)
		}
	}
	next := 256
	for value := range result {
		if !used[value] {
			result[value] = rune(next)
			next++
		}
	}
	return result
}
