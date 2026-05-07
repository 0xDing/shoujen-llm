# Shoujen LLM

Shoujen is a personal, interest-driven toy LLM project: a small Chinese-first
causal LM built on top of the `transformers` library. It combines a CJK-aware
tokenizer, an RWKV7 + attention hybrid backbone, Moonshot-style block
residuals, and Gemma4-style Per-Layer Embeddings (PLE) into a compact model
that can still be trained from scratch on local hardware.

The goal is practical: make a small model that learns classical Chinese and
modern instruction-following behavior without giving up too much context length
or training throughput. It is the successor to
[shoujen-rnn](https://github.com/0xDing/shoujen-rnn), now using a modern
hybrid-recurrent transformer instead of a plain RNN.

## Training Data

The corpus is a mix of three languages, but **classical Chinese (文言文) is the dominant register**:

- Classical Chinese — pre-modern literature, philosophy, and poetry (the bulk of the corpus, sourced from public domain collections).
- Modern Chinese (现代汉语) — Wikipedia, instruction-style data, forum threads, translated text.
- English — mostly via Chinese↔English parallel translation pairs, rendered into rotating templates.

The mix is intentional: classical Chinese gives the model a stylistic and semantic backbone (compact, dense, idiomatic), while modern Chinese and English keep the model usable as a chatbot and let it follow instructions in everyday language. Pretraining stages are organized as `s-init` → `s0` → `s1` → `s2` parquet shards under `data/processed/`.

## Tokenizer

The default tokenizer is
[`AgentBull/CJK-Tokenizer`](https://huggingface.co/AgentBull/CJK-Tokenizer),
loaded through the project wrapper:

```python
from shoujen.tokenizer import ShoujenTokenizer

tokenizer = ShoujenTokenizer.load()
```

It is a LLaMA tokenizer extended with single-token CJK characters. CJK text is
mostly character-level, while Latin text, punctuation, and other scripts keep
the original LLaMA subword behavior. Shoujen adds only the chat boundary tokens
`<im_start>` and `<im_end>` at load time; there is no dedicated pad token in
the default tokenizer path, so padding-only code falls back to EOS by position.

All training and inference scripts accept `--tokenizer` with a Hugging Face
tokenizer id, Hugging Face URL, or saved tokenizer directory. The old
`data/vocab.json` character-tokenizer path has been removed; passing a single
local vocab JSON file is an error.

### Tokenizer design trade-offs

**Why CJK character tokens instead of a locally trained BPE / SentencePiece?**

- *For classical Chinese, characters already are the morphemes.* A single 字 is a meaningful unit, often a whole word in pre-Tang prose. Keeping CJK characters atomic avoids merge rules biased toward modern high-frequency phrases.
- *Cross-register robustness.* The same model needs to read 《左傳》 and a modern Wikipedia paragraph in the same forward pass. CJK character tokens keep the core unit stable across registers, while LLaMA subwords keep English and code from exploding to byte- or character-level lengths.
- *No local tokenizer build step.* Training and inference load the same
  published tokenizer id, so the repo no longer carries or builds
  `data/vocab.json`.

**The cost** is that Chinese text is still close to character length. That is
acceptable for this Chinese-dominant model, and the inherited LLaMA tokenizer
keeps English-heavy spans more compact than the old pure character tokenizer.

## Model

Text-only causal LM. Architecturally it starts from the Qwen3.5-Hybrid idea,
then swaps the recurrent slots to [RWKV7](https://github.com/blinkdl/rwkv-lm)
and adds two depth-scaling aids: Block Attention Residuals and Gemma4-style
PLE. The result is a 114M-effective-parameter model whose stored parameter
count is intentionally embedding-heavy.

- `hidden_size = 512`
- `num_hidden_layers = 24`
- `layer_sharing = false`
- layout: `[RWKV7, RWKV7, RWKV7, Attention] × 6`
- `num_rwkv7_layers = 18`
- `num_attention_layers = 6`
- attention heads = 10, KV heads = 2 (GQA)
- `head_dim = 64`
- FFN = SwiGLU, `intermediate_size = 2048`
- norm = RMSNorm
- residual = Block Attention Residuals, `attnres_n_blocks = 4`
- Per-Layer Embeddings (Gemma4-style PLE), `hidden_size_per_layer_input = 256`
- token embedding and LM head tied
- `max_seq_len = 2048`
- default tokenizer vocab = 47,871
- parameters = 114.1M effective, 432.7M with embeddings

### Model design trade-offs

**RWKV7 + Attention Hybrid (3:1 Ratio).**
- Pure attention is O(T²) and bandwidth-bound on the M1 Ultra; pure RWKV is O(T) but loses something on tasks that need precise long-range pointer-style retrieval. The 3:1 mix keeps most layers cheap (RWKV7 has linear cost in sequence length and tiny KV state at inference) while giving the model 6 full attention layers spread across depth for genuine random-access lookups. This is the same intuition behind Qwen3.5-Hybrid and Jamba; we just slot RWKV7 in where they use Gated DeltaNet, since RWKV7 is somewhat better-validated at small scale and has reference kernels readily available.
- *Why RWKV7 specifically?* It supports a state-passing recurrence that converts cleanly to a closed-form parallel scan during training — so we get O(T) inference and parallel training without the engineering cost of writing a custom Triton kernel from scratch.

**24 layers at hidden 512 instead of e.g. 12 × 1024.**
- For a fixed parameter budget, deeper-and-narrower wins on language modeling perplexity in this effective size class (~100M), and the FFN cost (which dominates with `intermediate_size=2048`) scales with `hidden_size`, so going narrow keeps the per-step cost manageable on MPS. The downside is more sequential layers, i.e. less parallelism per token — acceptable for a single-GPU toy run.

**GQA (10 query heads, 2 KV heads).**
- 5:1 GQA cuts the KV cache by 5× at inference for a ~negligible quality hit at this scale. Critical because the attention layers are the only place we pay sequence-length-quadratic memory; shrinking KV is the single largest knob for context-window headroom on the M1.

**Block Attention Residuals (Moonshot-style).**
- Standard pre-norm transformers exhibit "residual stream takeover" at depth — later layers struggle to overwrite low-frequency directions deposited early. Block residuals every 4 layers act as a periodic refresh that lets later attention blocks read a less-saturated stream. Cheap to add, measurably helps with deep-and-narrow shapes.

**Gemma4-style Per-Layer Embeddings (PLE).**
- Every layer gets a token-identity side input instead of relying only on the
  residual stream to remember the original token. The PLE table is packed as
  `vocab_size × num_hidden_layers × hidden_size_per_layer_input`. With the
  default 47,871-token CJK tokenizer and `24 × 256` per-token PLE slots, this
  adds 294.1M lookup parameters. Following Gemma's parameter-count convention,
  the effective count excludes embedding lookup tables, while the "with
  embeddings" count includes the tied token embedding and PLE table.

**Tied embeddings + RMSNorm + RoPE.**
- Standard small-model defaults; RoPE because the attention layers need positional information that RWKV layers can't supply implicitly across the full window.

## Training

Hardware tested on:
- Apple M1 Ultra, PyTorch MPS.
- Nvidia RTX PRO 6000 WK, PyTorch CUDA.

Optimizer:
- Muon for hidden 2D matrices.
- AdamW for embeddings, norms, biases, gates, and small parameter groups.

Training objectives:
- Primary loss: next-token cross-entropy.
- SFT stage: assistant-only loss (mask provided by the chat-template encoder).
- Auxiliary loss: [Semantic Tube Prediction](https://github.com/galilai-group/llm-jepa#stp) is reserved for a future SFT path where message-local spans are available. It is not applied during packed pretraining, because random STP triples must not cross unrelated documents.

### Real-Tokenizer Learning Signal

For a quick architecture check, the benchmark script tokenizes TinyShakespeare
with the real default CJK tokenizer and feeds the same sampled token windows to
Shoujen and GPT-2 small. That keeps the production tokenizer, large vocabulary,
and PLE table in the loop.

```bash
uv run python scripts/benchmark_tinyshakespeare.py \
  --tokenizer-mode hf \
  --models shoujen gpt2-small \
  --data data/tinyshakespeare/input.txt \
  --batch-size 2 \
  --block-size 128 \
  --max-steps 100 \
  --eval-batches 4 \
  --optimizer-mode same-adamw \
  --adamw-lr 6e-4 \
  --gpt2-lr 6e-4 \
  --no-amp
```

On MPS, the 100-step run shows a stronger early loss drop for Shoujen:

| model | parameter count | initial val | final val | delta |
| --- | ---: | ---: | ---: | ---: |
| Shoujen | 114.1M effective / 432.7M with embeddings | 10.855 | 5.297 | 5.558 |
| GPT-2 small | 121.9M with embeddings | 10.932 | 6.525 | 4.406 |

Both models use the same token stream and AdamW setup. The comparison is most
useful as an early-learning signal under the tokenizer Shoujen is designed for:
the hybrid + PLE architecture reaches lower validation loss in the first 100
updates. Parameter count follows Gemma's convention: Shoujen is 114.1M
effective parameters, while its 432.7M stored parameters are dominated by the
294.1M PLE lookup table.

### Offline tokenized packed pretraining

After building and deduplicating `data/processed-clean/*.parquet`, the text
shards can be tokenized and document-packed once so training workers read ready
blocks instead of running the tokenizer:

```bash
uv run python scripts/build_packed_tokenized_corpus.py \
  --source-dir data/processed-clean \
  --out-dir data/processed-packed \
  --block-size 2048 \
  --tokenizer AgentBull/CJK-Tokenizer \
  --tokenizer-workers 8
```

The builder treats each source parquet row as an independent document, appends
EOS to that document, and emits fixed-size parquet rows containing
`block_size + 1` aligned arrays:

- `token_ids`: full next-token windows
- `seq_ids`: packed document ids, normalized per output row
- `position_ids`: per-document token positions, reset at document starts
- `sequence_starts`: reset flags for recurrent state

Rows may contain multiple documents, but training remains boundary-safe:
attention is masked to tokens with the same `seq_id`, RWKV state is reset at
`sequence_starts`, and labels that would predict across documents are replaced
with `-100`. The final partial tail of each shard is dropped unless it reaches
`block_size + 1` tokens.

Then train from the prepacked shards:

```bash
uv run python scripts/train_staged_packed.py \
  --data-dir data/processed-packed \
  --data-format packed \
  --block-size 2048
```

`--data-format auto` detects these shards from the parquet columns and switches
to the offline reader. Passing `--data-format packed` fails fast if a shard is
missing the required packed columns or its metadata block size does not match
the requested `--block-size`.

### Trainer-based packed pretraining

The staged parquet pretraining path also has a Transformers `Trainer` entrypoint
that streams text parquet rows and packs them online:

```bash
uv run python scripts/train_trainer_staged_packed.py \
  --data-dir data/processed-clean \
  --batch-size 3 \
  --gradient-accumulation-steps 1 \
  --no-gradient-checkpointing
```

On the local Apple M1 Max / 64GB probe with `block_size=2048`, fp16 AMP, and
real packed parquet short-run loss checks, the loss-aware default is:

- `batch_size=3`
- `gradient_accumulation_steps=1`
- `gradient_checkpointing=false`

The 64k-token batch-loss probe gave:

| candidate | val loss | tok/s | driver memory |
| --- | ---: | ---: | ---: |
| `batch=3, accum=1, checkpointing=false` | 8.7628 | 903 | 31.9GB |
| `batch=3, accum=1, checkpointing=true` | 8.7626 | 881 | 24.4GB |
| `batch=4, accum=1, checkpointing=false` | 8.9375 | 911 | 40.5GB |
| `batch=4, accum=2, checkpointing=false` | 9.1327 | 935 | 41.2GB |

Use `batch_size=4` only when peak throughput matters more than the short-run
loss signal. Use `batch_size=3 --gradient-checkpointing` when extra MPS memory
headroom matters. Do not use gradient accumulation by default; in the short
probe it increased the effective batch but did not improve loss.

### CUDA RWKV7 kernel

On CUDA, RWKV7 time-mixing now tries a runtime-compiled wind-backstepping
extension adapted from BlinkDL's RWKV-v7 reference CUDA kernel before falling
back to the transparent PyTorch recurrence. The first CUDA run compiles one
extension per `(dtype, head_dim, chunk_len)` combination and caches it under the
normal PyTorch extension cache.

Prerequisites:
- CUDA PyTorch with a matching local CUDA toolkit / `nvcc`
- `ninja` available to `torch.utils.cpp_extension`
- contiguous `[B, T, H, C]` tensors with `T` padded to the internal chunk length

Useful controls:
- `SHOUJEN_RWKV7_CUDA=0` disables the CUDA kernel path.
- `SHOUJEN_RWKV7_CUDA_VERBOSE=1` prints extension build logs.
- `uv run python scripts/test_rwkv7_cuda_equivalence.py` checks CUDA vs the
  PyTorch reference on a CUDA machine.

### Hyperparameter search

The formal staged pretraining path can be searched with:

```bash
uv run python scripts/hpo_train_staged_packed.py \
  --backend random \
  --n-trials 4 \
  --trial-steps 80 \
  --eval-every 40 \
  --eval-batches 8
```

For Optuna-backed search, install the optional HPO dependency first:

```bash
uv sync --extra hpo
uv run python scripts/hpo_train_staged_packed.py \
  --backend optuna \
  --n-trials 20 \
  --trial-steps 300 \
  --eval-every 100 \
  --eval-batches 20
```

The search reinitializes the model for every trial, samples staged parquet
training shards, evaluates on `s0-val.parquet`, and ranks trials by a composite
objective: validation loss, token throughput, and stability penalties from
AMP choice, gradient clipping, loss volatility, and optional QK logit tracking.
Results are written under `runs/hpo-staged-packed/`, including `summary.json`
and `best_train_command.txt` for the full training run with the selected
parameters.

## References

- https://github.com/MoonshotAI/Attention-Residuals
- https://arxiv.org/abs/2602.22617
- https://arxiv.org/abs/2502.16982
- https://github.com/0xDing/shoujen-rnn — the RNN-era predecessor

## License

Apache License 2.0. See [LICENSE](LICENSE).
