package tokenizer

import (
	"reflect"
	"testing"
)

func TestQwen3ChatGoldenConversation(t *testing.T) {
	got, err := (Qwen3Chat{DisableThinking: true}).Render([]Message{
		{Role: System, Content: "You are a concise support assistant."},
		{Role: User, Content: "Where is my refund?"},
		{Role: Assistant, Content: "Your refund is being processed."},
		{Role: User, Content: "Thanks. Can I change my email?"},
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nYou are a concise support assistant.<|im_end|>\n" +
		"<|im_start|>user\nWhere is my refund?<|im_end|>\n" +
		"<|im_start|>assistant\nYour refund is being processed.<|im_end|>\n" +
		"<|im_start|>user\nThanks. Can I change my email?<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if got != want {
		t.Fatalf("rendered = %q", got)
	}
}

func TestByteLevelGoldenEdgeCases(t *testing.T) {
	tokenizer := goldenByteLevelTokenizer(t)
	cases := []struct {
		name string
		text string
		ids  []int
	}{
		{name: "ascii", text: "Hi!", ids: []int{4, 3, 9}},
		{name: "newline", text: "\n", ids: []int{0, 10}},
		{name: "special", text: "Hi<|im_end|>", ids: []int{4, 3, 12}},
	}
	for _, tc := range cases {
		got, err := tokenizer.Encode(tc.text)
		if err != nil {
			t.Fatalf("%s encode: %v", tc.name, err)
		}
		if !reflect.DeepEqual(got, tc.ids) {
			t.Fatalf("%s ids = %v, want %v", tc.name, got, tc.ids)
		}
		decoded, err := tokenizer.Decode(got, false)
		if err != nil || decoded != " "+tc.text {
			t.Fatalf("%s decoded = %q, err = %v", tc.name, decoded, err)
		}
	}
}

func TestByteLevelDecodeSkipsSpecialTokensGolden(t *testing.T) {
	tokenizer := goldenByteLevelTokenizer(t)
	got, err := tokenizer.Decode([]int{4, 3, 12, 13}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got != " Hi<domain>" {
		t.Fatalf("decoded = %q", got)
	}
}

func goldenByteLevelTokenizer(t *testing.T) *ByteLevelTokenizer {
	t.Helper()
	tokenizer, err := LoadByteLevelTokenizer(writeTokenizerJSON(t, `{
  "model":{"type":"BPE","vocab":{"\u0120":0,"\u0120\u0120":1,"H":2,"i":3,"\u0120H":4,"!":9,"\u010a":10},"merges":[["\u0120","H"],["\u0120","\u0120"]]},
  "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":true},
  "decoder":{"type":"ByteLevel"},
  "added_tokens":[{"id":12,"content":"<|im_end|>","special":true},{"id":13,"content":"<domain>","special":false}]
}`))
	if err != nil {
		t.Fatal(err)
	}
	return tokenizer
}
