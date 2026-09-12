package tokenizer

import (
	"strings"
	"unicode"
)

// ByteLevel splits text using the GPT-2 ByteLevel pre-tokenization rules.
type ByteLevel struct {
	// AddPrefixSpace prepends one ASCII space when text does not start with one.
	AddPrefixSpace bool
}

// Split returns raw text pieces to pass individually through ByteLevelBPE.
func (p ByteLevel) Split(text string) []string {
	if text == "" {
		return nil
	}
	if p.AddPrefixSpace && text[0] != ' ' {
		text = " " + text
	}
	chars := []rune(text)
	pieces := make([]string, 0, len(chars)/2+1)
	for start := 0; start < len(chars); {
		if end := contractionEnd(chars, start); end > start {
			pieces = append(pieces, string(chars[start:end]))
			start = end
			continue
		}
		end := start
		if chars[start] == ' ' && start+1 < len(chars) {
			end = prefixedClassEnd(chars, start, unicode.IsLetter)
			if end == start {
				end = prefixedClassEnd(chars, start, unicode.IsNumber)
			}
			if end == start {
				end = prefixedSymbolEnd(chars, start)
			}
		}
		if end == start {
			switch {
			case unicode.IsLetter(chars[start]):
				end = classEnd(chars, start, unicode.IsLetter)
			case unicode.IsNumber(chars[start]):
				end = classEnd(chars, start, unicode.IsNumber)
			case !unicode.IsSpace(chars[start]):
				end = symbolEnd(chars, start)
			default:
				end = whitespaceEnd(chars, start)
			}
		}
		pieces = append(pieces, string(chars[start:end]))
		start = end
	}
	return pieces
}

// Encode splits text and encodes each piece with model.
func (p ByteLevel) Encode(model Encoder, text string) ([]int, error) {
	if model == nil {
		return nil, ErrInvalidByteLevel
	}
	pieces := p.Split(text)
	ids := make([]int, 0, len(text))
	for _, piece := range pieces {
		encoded, err := model.EncodePiece(piece)
		if err != nil {
			return nil, err
		}
		ids = append(ids, encoded...)
	}
	return ids, nil
}

func contractionEnd(chars []rune, start int) int {
	if chars[start] != '\'' {
		return start
	}
	for _, suffix := range []string{"re", "ve", "ll", "s", "t", "m", "d"} {
		end := start + 1 + len(suffix)
		if end <= len(chars) && strings.EqualFold(string(chars[start+1:end]), suffix) {
			return end
		}
	}
	return start
}

func prefixedClassEnd(chars []rune, start int, matches func(rune) bool) int {
	if !matches(chars[start+1]) {
		return start
	}
	return classEnd(chars, start+1, matches)
}

func classEnd(chars []rune, start int, matches func(rune) bool) int {
	end := start
	for end < len(chars) && matches(chars[end]) {
		end++
	}
	return end
}

func prefixedSymbolEnd(chars []rune, start int) int {
	if !isSymbol(chars[start+1]) {
		return start
	}
	return symbolEnd(chars, start+1)
}

func symbolEnd(chars []rune, start int) int {
	end := start
	for end < len(chars) && isSymbol(chars[end]) {
		end++
	}
	return end
}

func isSymbol(value rune) bool {
	return !unicode.IsSpace(value) && !unicode.IsLetter(value) && !unicode.IsNumber(value)
}

func whitespaceEnd(chars []rune, start int) int {
	end := start
	for end < len(chars) && unicode.IsSpace(chars[end]) {
		end++
	}
	if end < len(chars) && end-start > 1 {
		return end - 1
	}
	return end
}
