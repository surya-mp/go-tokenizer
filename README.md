# go-tokenizer

Tokenizer and chat-template contracts for local SLM and LLM tooling.

The initial implementation provides generic messages, ChatML rendering, BPE,
and standard GPT-2 ByteLevel splitting plus byte encoding/decoding.
`LoadByteLevelTokenizer` loads the strict BPE/ByteLevel tokenizer.json subset
and added tokens. `LoadByteLevelTokenizerAssets` also resolves configured
BOS/EOS/PAD/UNK IDs. `Qwen3Chat` supports Qwen3's text-only template branch.
Normalizers, post-processors, tools, and other plugins fail explicitly until
they have golden vectors.
