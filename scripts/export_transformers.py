"""Export a legacy Shoujen .pt checkpoint to a Transformers model directory.

Examples:
    uv run python scripts/export_transformers.py \
        --ckpt runs/sft-packed/last.pt \
        --output runs/sft-packed-hf

    uv run python scripts/export_transformers.py \
        --ckpt runs/sft-packed/last.pt \
        --output runs/sft-packed-hf \
        --tokenizer AgentBull/CJK-Tokenizer \
        --verify
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shoujen.config import ShoujenConfig  # noqa: E402
from shoujen.model import ShoujenForCausalLM  # noqa: E402
from shoujen.tokenizer import DEFAULT_TOKENIZER_ID, ShoujenTokenizer  # noqa: E402


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/sft-packed/last.pt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional config.json override. Defaults to the config embedded in --ckpt.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_ID,
        help="HF tokenizer id/URL or saved tokenizer directory to copy into the output.",
    )
    parser.add_argument("--no-tokenizer", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=["keep", *DTYPES.keys()],
        default="keep",
        help="Optional dtype conversion before saving safetensors.",
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--auto-class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write auto_map metadata/code so AutoModelForCausalLM.from_pretrained(..., "
            "trust_remote_code=True) can load the export when the shoujen package is importable."
        ),
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace, state: dict[str, Any]) -> ShoujenConfig:
    if args.config is not None:
        return ShoujenConfig.from_json(args.config)

    config_data = state.get("config")
    if not isinstance(config_data, dict):
        raise SystemExit(f"{args.ckpt} does not contain a checkpoint config; pass --config")
    return ShoujenConfig(**dict(config_data))


def load_legacy_model(args: argparse.Namespace) -> tuple[ShoujenForCausalLM, ShoujenConfig, dict[str, Any]]:
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "model" not in state:
        raise SystemExit(f"{args.ckpt} does not contain a 'model' state_dict")

    config = load_config(args, state)
    model = ShoujenForCausalLM(config)
    model.load_state_dict(state["model"])
    model.eval()

    if args.dtype != "keep":
        model = model.to(dtype=DTYPES[args.dtype])
        model.config.dtype = str(DTYPES[args.dtype]).replace("torch.", "")

    return model, config, state


def save_tokenizer(args: argparse.Namespace, config: ShoujenConfig) -> None:
    if args.no_tokenizer:
        return

    tokenizer = ShoujenTokenizer.load(args.tokenizer)
    if tokenizer.vocab_size != config.vocab_size:
        raise SystemExit(
            f"Tokenizer vocab size {tokenizer.vocab_size} does not match model vocab size "
            f"{config.vocab_size}. Pass the tokenizer used for training, or use --no-tokenizer."
        )
    tokenizer.save(args.output)


def write_export_metadata(args: argparse.Namespace, state: dict[str, Any]) -> None:
    metadata = {
        "source_checkpoint": str(args.ckpt),
        "source_step": state.get("step"),
        "format": "transformers-pretrained-safetensors",
    }
    (args.output / "shoujen_export.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_auto_code_standalone(output: Path) -> None:
    model_py = output / "model.py"
    if model_py.exists():
        text = model_py.read_text(encoding="utf-8")
        text = text.replace("from shoujen.config import ShoujenConfig", "from .config import ShoujenConfig")
        text = text.replace("from shoujen.modules.norm import RMSNorm", "from .norm import RMSNorm")
        text = text.replace("from shoujen.modules.rwkv7 import RWKV7TimeMix", "from .rwkv7 import RWKV7TimeMix")
        if "from .rwkv7_cuda import can_use_cuda_wind_backstepping as _shoujen_cuda_probe" not in text:
            text = text.replace(
                "from .rwkv7 import RWKV7TimeMix\n",
                (
                    "from .rwkv7 import RWKV7TimeMix\n"
                    "from .rwkv7_cuda import can_use_cuda_wind_backstepping as _shoujen_cuda_probe\n"
                    "from .rwkv7_mps import can_use_mps_wind_backstepping as _shoujen_mps_probe\n"
                ),
            )
        model_py.write_text(text, encoding="utf-8")

    modules_dir = REPO_ROOT / "shoujen" / "modules"
    for name in ("norm.py", "rwkv7.py", "rwkv7_cuda.py", "rwkv7_mps.py"):
        shutil.copy2(modules_dir / name, output / name)

    rwkv7_py = output / "rwkv7.py"
    text = rwkv7_py.read_text(encoding="utf-8")
    text = text.replace(
        "from shoujen.modules.rwkv7_cuda import can_use_cuda_wind_backstepping, cuda_wind_backstepping",
        "from .rwkv7_cuda import can_use_cuda_wind_backstepping, cuda_wind_backstepping",
    )
    text = text.replace(
        "from shoujen.modules.rwkv7_mps import can_use_mps_wind_backstepping, mps_wind_backstepping",
        "from .rwkv7_mps import can_use_mps_wind_backstepping, mps_wind_backstepping",
    )
    rwkv7_py.write_text(text, encoding="utf-8")


def verify_export(args: argparse.Namespace) -> None:
    if args.auto_class:
        model = AutoModelForCausalLM.from_pretrained(args.output, trust_remote_code=True)
    else:
        model = ShoujenForCausalLM.from_pretrained(args.output)
    model.eval()

    if not args.no_tokenizer:
        AutoTokenizer.from_pretrained(args.output, trust_remote_code=True)
    print(f"verified load: {model.__class__.__name__}", flush=True)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model, config, state = load_legacy_model(args)
    if args.auto_class:
        config.register_for_auto_class("AutoConfig")
        ShoujenForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    if args.auto_class:
        make_auto_code_standalone(args.output)
    save_tokenizer(args, config)
    write_export_metadata(args, state)

    print(f"saved Transformers model to {args.output}", flush=True)
    print(f"checkpoint step={state.get('step')} dtype={model.dtype}", flush=True)
    if args.verify:
        verify_export(args)


if __name__ == "__main__":
    main()
