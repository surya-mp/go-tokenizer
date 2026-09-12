package tokenizer

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestBPEMergesByRank(t *testing.T) {
	model, err := NewBPE(
		map[string]int{"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hell": 6, "hello": 7},
		[][2]string{{"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}}, "",
	)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := model.EncodePiece("hello")
	if err != nil || !reflect.DeepEqual(ids, []int{7}) {
		t.Fatalf("ids = %v, err = %v", ids, err)
	}
	text, err := model.Decode(ids)
	if err != nil || text != "hello" {
		t.Fatalf("text = %q, err = %v", text, err)
	}
}

func TestBPEUnknownToken(t *testing.T) {
	model, err := NewBPE(map[string]int{"a": 0}, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	_, err = model.EncodePiece("b")
	if !errors.Is(err, ErrUnknownToken) {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadBPE(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	data := []byte("{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0,\"b\":1,\"ab\":2,\"<unk>\":3},\"merges\":[[\"a\",\"b\"]],\"unk_token\":\"<unk>\"}}")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	model, err := LoadBPE(path)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := model.EncodePiece("ab")
	if err != nil || !reflect.DeepEqual(ids, []int{2}) {
		t.Fatalf("ids = %v, err = %v", ids, err)
	}
	ids, err = model.EncodePiece("z")
	if err != nil || !reflect.DeepEqual(ids, []int{3}) {
		t.Fatalf("unknown IDs = %v, err = %v", ids, err)
	}
}
