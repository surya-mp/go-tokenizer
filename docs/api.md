# API reference

Canonical API documentation is generated from Go doc comments on
[pkg.go.dev](https://pkg.go.dev/github.com/surya-mp/go-tokenizer).

| API | Use |
| --- | --- |
| `LoadByteLevelTokenizerAssets` | Load `tokenizer.json` plus configured special IDs. |
| `LoadByteLevelTokenizer` | Load the supported strict tokenizer JSON subset. |
| `ByteLevelTokenizer.Encode` / `Decode` | Convert text and token IDs. |
| `Qwen3Chat.Render` | Render Qwen3 text chat prompts. |
| `ChatML.Render` | Render generic ChatML prompts. |
| `NewBPE` / `NewByteLevelBPE` | Build a tokenizer from explicit vocabulary and merges. |

Unsupported tokenizer behavior returns a typed package error rather than
silently changing tokenization.
