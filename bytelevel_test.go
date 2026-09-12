package tokenizer

import (
	"errors"
	"reflect"
	"testing"
)

func TestByteLevelBPERoundTrip(t *testing.T) {
	bpe, err := NewBPE(
		map[string]int{"Ġ": 0, "h": 1, "i": 2, "Ġh": 3, "Ġhi": 4, "Ã": 5, "©": 6},
		[][2]string{{"Ġ", "h"}, {"Ġh", "i"}}, "",
	)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewByteLevelBPE(bpe)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := model.EncodePiece(" hié")
	if err != nil || !reflect.DeepEqual(ids, []int{4, 5, 6}) {
		t.Fatalf("ids = %v, err = %v", ids, err)
	}
	text, err := model.Decode(ids)
	if err != nil || text != " hié" {
		t.Fatalf("text = %q, err = %v", text, err)
	}
}

func TestByteLevelBPERejectsNonByteToken(t *testing.T) {
	bpe, err := NewBPE(map[string]int{"🙂": 0}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewByteLevelBPE(bpe)
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.Decode([]int{0})
	if !errors.Is(err, ErrInvalidByteLevel) {
		t.Fatalf("err = %v", err)
	}
}
