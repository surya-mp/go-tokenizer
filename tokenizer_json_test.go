package tokenizer

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadByteLevelTokenizerAddedTokens(t *testing.T) {
	path := writeTokenizerJSON(t, `{
  "model":{"type":"BPE","vocab":{"Ġ":0,"a":1,"b":2,"Ġa":3},"merges":[["Ġ","a"]]},
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":true},
  "decoder":{"type":"ByteLevel"},
  "added_tokens":[{"id":4,"content":"<eos>","special":true}]
}`)
	tokenizer, err := LoadByteLevelTokenizer(path)
	if err != nil {
		t.Fatal(err)
	}
	ids, err := tokenizer.Encode("a<eos>b")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ids, []int{3, 4, 2}) {
		t.Fatalf("ids = %v", ids)
	}
	decoded, err := tokenizer.Decode(ids, false)
	if err != nil || decoded != " a<eos>b" {
		t.Fatalf("decoded = %q, err = %v", decoded, err)
	}
	decoded, err = tokenizer.Decode(ids, true)
	if err != nil || decoded != " ab" {
		t.Fatalf("skipped = %q, err = %v", decoded, err)
	}
	if id, ok := tokenizer.AddedTokenID("<eos>"); !ok || id != 4 {
		t.Fatalf("added token = %d, %t", id, ok)
	}
}

func TestLoadByteLevelTokenizerRejectsComposition(t *testing.T) {
	path := writeTokenizerJSON(t, `{
  "model":{"type":"BPE","vocab":{"a":0},"merges":[]},
  "pre_tokenizer":{"type":"Whitespace"},"decoder":{"type":"ByteLevel"}
}`)
	_, err := LoadByteLevelTokenizer(path)
	if !errors.Is(err, ErrUnsupportedTokenizerJSON) {
		t.Fatalf("err = %v", err)
	}
}

func TestLoadByteLevelTokenizerRejectsIgnoredBehavior(t *testing.T) {
	for _, field := range []string{
		`"normalizer":{"type":"Lowercase"}`,
		`"post_processor":{"type":"TemplateProcessing"}`,
		`"added_tokens":[{"id":1,"content":"<x>","lstrip":true}]`,
	} {
		path := writeTokenizerJSON(t, `{
  "model":{"type":"BPE","vocab":{"a":0},"merges":[]},
  "pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},`+field+`
}`)
		_, err := LoadByteLevelTokenizer(path)
		if !errors.Is(err, ErrUnsupportedTokenizerJSON) && !errors.Is(err, ErrInvalidTokenizerJSON) {
			t.Fatalf("field %s err = %v", field, err)
		}
	}
}

func TestLoadByteLevelTokenizerAssetsResolvesSpecialIDs(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(`{
  "model":{"type":"BPE","vocab":{"Ġ":0,"a":1},"merges":[]},
  "pre_tokenizer":{"type":"ByteLevel"},"decoder":{"type":"ByteLevel"},
  "added_tokens":[{"id":2,"content":"<s>","special":true},{"id":3,"content":"</s>","special":true}]
}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), []byte(`{
  "bos_token":{"content":"<s>"},"eos_token":"</s>","pad_token":"a"
}`), 0o600); err != nil {
		t.Fatal(err)
	}
	_, ids, err := LoadByteLevelTokenizerAssets(dir)
	if err != nil {
		t.Fatal(err)
	}
	if ids != (SpecialTokenIDs{BOS: 2, EOS: 3, PAD: 1, UNK: -1}) {
		t.Fatalf("special IDs = %#v", ids)
	}
}

func writeTokenizerJSON(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
