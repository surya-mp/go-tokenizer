package tokenizer

import (
	"errors"
	"testing"
)

func TestChatMLRender(t *testing.T) {
	text, err := (ChatML{}).Render([]Message{{Role: System, Content: "help"}, {Role: User, Content: "refund"}}, true)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\nhelp<|im_end|>\n<|im_start|>user\nrefund<|im_end|>\n<|im_start|>assistant\n"
	if text != want {
		t.Fatalf("text = %q", text)
	}
}

func TestQwen3ChatTextOnlyTemplate(t *testing.T) {
	rendered, err := (Qwen3Chat{}).Render([]Message{
		{Role: User, Content: "first"}, {Role: Assistant, Content: "answer"}, {Role: User, Content: "second"},
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>user\nfirst<|im_end|>\n" +
		"<|im_start|>assistant\nanswer<|im_end|>\n" +
		"<|im_start|>user\nsecond<|im_end|>\n<|im_start|>assistant\n"
	if rendered != want {
		t.Fatalf("rendered = %q", rendered)
	}
}

func TestQwen3ChatFinalAssistantAndDisabledThinking(t *testing.T) {
	rendered, err := (Qwen3Chat{DisableThinking: true}).Render([]Message{
		{Role: User, Content: "question"}, {Role: Assistant, Content: "\nanswer"},
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>user\nquestion<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n" +
		"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if rendered != want {
		t.Fatalf("rendered = %q", rendered)
	}
}

func TestChatMLRejectsRole(t *testing.T) {
	_, err := (ChatML{}).Render([]Message{{Role: "tool", Content: "x"}}, false)
	if !errors.Is(err, ErrInvalidMessage) {
		t.Fatalf("err = %v", err)
	}
}
