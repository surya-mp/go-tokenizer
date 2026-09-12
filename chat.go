// Package tokenizer defines tokenizer and supported chat-rendering contracts.
package tokenizer

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrInvalidMessage reports an unsupported message role or invalid content.
	ErrInvalidMessage = errors.New("tokenizer: invalid message")
	// ErrUnsupportedTemplate reports a chat template without a registered renderer.
	ErrUnsupportedTemplate = errors.New("tokenizer: unsupported chat template")
)

// Role identifies a chat message participant.
type Role string

const (
	// System is the system-instruction role.
	System Role = "system"
	// User is the end-user role.
	User Role = "user"
	// Assistant is the model-response role.
	Assistant Role = "assistant"
)

// Message is one structured chat turn.
type Message struct {
	// Role identifies the message participant.
	Role Role
	// Content is the message text.
	Content string
}

// Renderer serializes structured messages into model input text.
type Renderer interface {
	// Render serializes messages and optionally opens an assistant turn.
	Render(messages []Message, addGenerationPrompt bool) (string, error)
}

// ChatML renders the common start-role-content-end chat format.
type ChatML struct {
	// StartToken defaults to <|im_start|>.
	StartToken string
	// EndToken defaults to <|im_end|>.
	EndToken string
}

// Qwen3Chat renders the text-only branch of Qwen3's published chat template.
// Tool calls and structured reasoning require a richer message contract.
type Qwen3Chat struct {
	// DisableThinking emits Qwen3's empty think block in a generation prompt.
	DisableThinking bool
}

// Render serializes system, user, and assistant text messages for Qwen3.
func (q Qwen3Chat) Render(messages []Message, addGenerationPrompt bool) (string, error) {
	lastQuery := len(messages) - 1
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == User {
			lastQuery = index
			break
		}
	}
	var result strings.Builder
	for index, message := range messages {
		if message.Role != System && message.Role != User && message.Role != Assistant {
			return "", fmt.Errorf("%w: %q", ErrInvalidMessage, message.Role)
		}
		result.WriteString("<|im_start|>")
		result.WriteString(string(message.Role))
		result.WriteByte('\n')
		if message.Role == Assistant && index > lastQuery {
			result.WriteString("<think>\n\n</think>\n\n")
			result.WriteString(strings.TrimLeft(message.Content, "\n"))
		} else {
			result.WriteString(message.Content)
		}
		result.WriteString("<|im_end|>\n")
	}
	if addGenerationPrompt {
		result.WriteString("<|im_start|>assistant\n")
		if q.DisableThinking {
			result.WriteString("<think>\n\n</think>\n\n")
		}
	}
	return result.String(), nil
}

// Render serializes messages. When addGenerationPrompt is true, it starts an
// assistant turn without adding an end token.
func (c ChatML) Render(messages []Message, addGenerationPrompt bool) (string, error) {
	start, end := c.StartToken, c.EndToken
	if start == "" {
		start = "<|im_start|>"
	}
	if end == "" {
		end = "<|im_end|>"
	}
	var result strings.Builder
	for _, message := range messages {
		if message.Role != System && message.Role != User && message.Role != Assistant {
			return "", fmt.Errorf("%w: %q", ErrInvalidMessage, message.Role)
		}
		result.WriteString(start)
		result.WriteString(string(message.Role))
		result.WriteByte('\n')
		result.WriteString(message.Content)
		result.WriteString(end)
		result.WriteByte('\n')
	}
	if addGenerationPrompt {
		result.WriteString(start)
		result.WriteString(string(Assistant))
		result.WriteByte('\n')
	}
	return result.String(), nil
}
