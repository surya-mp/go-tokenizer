package tokenizer

import (
	"reflect"
	"testing"
)

func TestByteLevelSplit(t *testing.T) {
	cases := []struct {
		text string
		want []string
	}{
		{"Hello,  world!\n", []string{"Hello", ",", " ", " world", "!", "\n"}},
		{"I'm 42", []string{"I", "'m", " 42"}},
		{"  !", []string{" ", " !"}},
	}
	for _, test := range cases {
		if got := (ByteLevel{}).Split(test.text); !reflect.DeepEqual(got, test.want) {
			t.Errorf("Split(%q) = %#v, want %#v", test.text, got, test.want)
		}
	}
}

func TestByteLevelPrefixSpaceAndEncode(t *testing.T) {
	bpe, err := NewBPE(map[string]int{"Ġ": 0, "h": 1, "i": 2, "Ġh": 3, "Ġhi": 4}, [][2]string{{"Ġ", "h"}, {"Ġh", "i"}}, "")
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewByteLevelBPE(bpe)
	if err != nil {
		t.Fatal(err)
	}
	pre := ByteLevel{AddPrefixSpace: true}
	if got := pre.Split("hi"); !reflect.DeepEqual(got, []string{" hi"}) {
		t.Fatalf("pieces = %#v", got)
	}
	ids, err := pre.Encode(model, "hi")
	if err != nil || !reflect.DeepEqual(ids, []int{4}) {
		t.Fatalf("ids = %v, err = %v", ids, err)
	}
}

func TestQwenByteLevelSplit(t *testing.T) {
	got := (ByteLevel{Qwen: true}).Split("!hello 12\n")
	want := []string{"!hello", " ", "1", "2", "\n"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("pieces = %#v, want %#v", got, want)
	}
}
