"""Run inference / chat with a trained ShoujenLM checkpoint.

Examples:
    # Single completion from a raw prompt
    python scripts/infer.py --ckpt runs/sft/last.pt --vocab data/vocab.json \
        --prompt "今天天气真好"

    # Interactive chat mode
    python scripts/infer.py --ckpt runs/sft/last.pt --vocab data/vocab.json --chat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shoujen.config import ShoujenConfig
from shoujen.generate import generate
from shoujen.model import ShoujenLM
from shoujen.tokenizer import ShoujenTokenizer
from shoujen.train_utils import pick_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--prompt", default=None)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--system", default=None, help="System prompt for chat mode")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_model(ckpt_path: str, device: torch.device) -> tuple[ShoujenLM, ShoujenConfig]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state["config"]
    if isinstance(cfg_dict.get("layer_pattern"), list):
        cfg_dict["layer_pattern"] = tuple(cfg_dict["layer_pattern"])
    config = ShoujenConfig(**cfg_dict)
    model = ShoujenLM(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config


def stream_generation(model, tokenizer, prompt_ids, args, device) -> str:
    pieces: list[int] = []
    for tok in generate(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    ):
        pieces.append(tok)
        # Decode incrementally. Byte-fallback tokens may need buffering, but
        # decode() handles that on the running buffer too — we just decode the
        # whole tail each time and print only the new text.
        text = tokenizer.decode(pieces, skip_special=True)
        sys.stdout.write(text[stream_generation._printed :])
        sys.stdout.flush()
        stream_generation._printed = len(text)
    print()
    return tokenizer.decode(pieces, skip_special=True)


stream_generation._printed = 0  # type: ignore[attr-defined]


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device={device}", flush=True)

    tokenizer = ShoujenTokenizer.load(args.vocab)
    model, _ = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} ({model.num_parameters() / 1e6:.2f}M params)", flush=True)

    if args.chat:
        history: list[dict] = []
        if args.system:
            history.append({"role": "system", "content": args.system})
        try:
            while True:
                user = input("user> ").strip()
                if not user:
                    continue
                if user in ("/quit", "/exit"):
                    break
                if user == "/reset":
                    history = ([{"role": "system", "content": args.system}] if args.system else [])
                    continue
                history.append({"role": "user", "content": user})
                prompt_ids, _ = tokenizer.encode_chat(history, add_generation_prompt=True)
                stream_generation._printed = 0
                print("assistant> ", end="", flush=True)
                reply = stream_generation(model, tokenizer, prompt_ids, args, device)
                history.append({"role": "assistant", "content": reply})
        except (EOFError, KeyboardInterrupt):
            print()
        return

    prompt = args.prompt or "你好"
    prompt_ids = tokenizer.encode(prompt)
    print(prompt, end="", flush=True)
    stream_generation._printed = 0
    stream_generation(model, tokenizer, prompt_ids, args, device)


if __name__ == "__main__":
    main()
