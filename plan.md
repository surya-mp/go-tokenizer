# Development plan

## Purpose

go-tokenizer loads and executes Hugging Face tokenizer assets and renders chat
templates. It is model- and application-domain-neutral. It returns text, token
IDs, offsets, and masks; it never creates model tensors.

## Public API goals

- Keep tokenizer core independent of transport. An optional hfhub convenience
  subpackage may resolve local assets without making the core depend on it.
- Define a Tokenizer interface for Encode, Decode, special token IDs, padding,
  truncation, and offsets.
- Define a Renderer interface for structured chat messages and an explicit
  generation-prompt option.
- Load tokenizer JSON, tokenizer configuration, special-token maps, and
  generation configuration from local files.
- Expose immutable tokenizer configuration and no mutable global state.
- Reject unsupported model, normalizer, pre-tokenizer, decoder, or template
  constructs with a typed error.

## Development milestones

### M1: metadata and renderer registry

- Complete tokenizer asset loading and typed validation.
- Keep ChatML as one renderer plugin, not the default behavior for all models.
- Define a renderer registry selected from explicit model metadata.
- Add message roles, tool-message extension points, stop tokens, and special
  token policy without application-specific prompts.
- Define exact truncation and padding semantics.

### M2: tokenizer-model plugins

- Implement the Hugging Face tokenizer JSON model contract.
- Completed: pre-tokenized BPE merge engine, BPE tokenizer JSON model load,
  standard ByteLevel byte encoding/decoding, GPT-2 ByteLevel splitting, and
  strict BPE/ByteLevel tokenizer.json loading with added-token handling and
  tokenizer-config special-token IDs. Unsupported normalizers, post-processors,
  and added-token matching flags are rejected rather than approximated.
- Next: tokenizer.json sequence composition and golden vectors.
- Implement added-token matching, byte fallback, normalizer composition, and
  offset mapping.
- Implement decoder composition and special-token skipping rules.
- Add WordPiece and Unigram as independently tested plugins only after BPE is
  correct.

### M3: template compatibility

- Completed: text-only Qwen3 renderer with explicit unsupported tool/reasoning
  branches.
- Implement a constrained, deterministic chat-template evaluator or a
  precompiled renderer plugin system.
- Support only template operations proven by golden vectors.
- Record template source and renderer version in output metadata.
- Never execute arbitrary template code or silently replace an unknown template.

## Test plan

- Golden fixtures from Python tokenizers for text, IDs, offsets, decode, padding,
  truncation, and special tokens.
- Golden fixtures for multiple public model families, beginning with one
  byte-level BPE model.
- Exact rendered-chat fixtures for system, user, assistant, empty content,
  multiline content, and generation prompt.
- Property tests for encode/decode invariants where normalization permits them.
- Fuzz malformed tokenizer JSON, oversized merges, invalid Unicode, and added
  token overlap.
- Race tests for concurrent immutable tokenizer use.

## Integration plan

- Load tokenizer assets from a go-hfhub snapshot.
- Pass rendered prompt token IDs and assistant-boundary IDs to go-sft.
- Use the same tokenizer instance for causal-LM generation decode.
- Verify template render and token IDs match the Python reference before a model
  training integration is accepted.

## Release criteria

- Every supported tokenizer/template pair has versioned golden vectors.
- Unsupported input fails explicitly with no approximate fallback.
- Encode and Decode allocation behavior is benchmarked for long prompts.
- Public docs list supported tokenizer models and chat renderers.

## Non-goals

- Model execution, vocabulary downloading, dataset loading, training loss, or
  web serving.
