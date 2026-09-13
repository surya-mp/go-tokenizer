# go-tokenizer

Tokenizer and chat-template contracts for local SLM and LLM tooling in Go.

## Install

```sh
go get github.com/surya-mp/go-tokenizer
```

See the [API reference](docs/api.md).

The initial implementation provides generic messages, ChatML rendering, BPE,
and standard GPT-2 ByteLevel splitting plus byte encoding/decoding.
`LoadByteLevelTokenizer` loads the strict BPE/ByteLevel tokenizer.json subset
and added tokens. `LoadByteLevelTokenizerAssets` also resolves configured
BOS/EOS/PAD/UNK IDs. `Qwen3Chat` supports Qwen3's text-only template branch.
Normalizers, post-processors, tools, and other plugins fail explicitly until
they have golden vectors.

```go
encoder, ids, err := tokenizer.LoadByteLevelTokenizerAssets(snapshotDir)
prompt, err := (tokenizer.Qwen3Chat{}).Render(messages, true)
tokenIDs, err := encoder.Encode(prompt)
_ = ids
_ = tokenIDs
_ = err
```

Use this package when you need a Go Hugging Face tokenizer subset, Qwen
tokenizer assets, Qwen chat template rendering, ChatML prompts, or local
encode/decode fixtures for supervised fine-tuning and causal language model
evaluation. Golden tests pin Qwen3 chat-template output and ByteLevel edge
cases including whitespace, special tokens, and newline handling.
