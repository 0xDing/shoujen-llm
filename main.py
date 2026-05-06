"""Project entry point. Sanity-checks the imports + tokenizer roundtrip.

Real usage flow (see README and scripts/ for details):
    python scripts/build_tokenizer.py --output data/vocab.json [--offline]
    python scripts/train.py  --mode pretrain ...
    python scripts/train.py  --mode sft      ...
    python scripts/infer.py  --ckpt runs/sft/last.pt --vocab data/vocab.json --chat
"""
from shoujen import ShoujenConfig, ShoujenLM, ShoujenTokenizer
from shoujen.tokenizer import default_charset


def main():
    print("Hello from shoujen-llm!")

    tok = ShoujenTokenizer.from_charset(default_charset())
    text = "Hello, 世界！"  # 世界 won't be in default_charset (no CJK), exercises byte fallback
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"vocab_size={tok.vocab_size}")
    print(f"text     : {text!r}")
    print(f"encoded  : {ids[:20]}{'...' if len(ids) > 20 else ''}")
    print(f"decoded  : {decoded!r}")
    assert decoded == text, "tokenizer roundtrip failed"

    cfg = ShoujenConfig(vocab_size=tok.vocab_size)
    print(f"default config: {cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}")
    print(f"layer types   : {cfg.shoujen_layer_types[:8]} ...")
    n = sum(p.numel() for p in ShoujenLM(cfg).parameters())
    print(f"model params  : {n / 1e6:.2f}M")


if __name__ == "__main__":
    main()
