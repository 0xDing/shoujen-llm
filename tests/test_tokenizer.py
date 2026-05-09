import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from shoujen.tokenizer import (
    IM_END_TOKEN,
    IM_START_TOKEN,
    SPECIAL_TOKENS,
    THINK_END_TOKEN,
    THINK_START_TOKEN,
    ShoujenTokenizer,
)


def _tiny_hf_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
    )


def test_local_vocab_json_is_not_supported(tmp_path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="vocab.json"):
        ShoujenTokenizer.load(vocab_path)


def test_project_special_tokens_include_think_tags() -> None:
    tokenizer = ShoujenTokenizer(_tiny_hf_tokenizer())

    assert SPECIAL_TOKENS == [
        IM_START_TOKEN,
        IM_END_TOKEN,
        THINK_START_TOKEN,
        THINK_END_TOKEN,
    ]
    for token in SPECIAL_TOKENS:
        assert token in tokenizer.hf_tokenizer.all_special_tokens
        assert tokenizer.hf_tokenizer.convert_tokens_to_ids(token) != tokenizer.hf_tokenizer.unk_token_id

    ids = tokenizer.encode(f"{THINK_START_TOKEN} reason {THINK_END_TOKEN}")
    assert tokenizer.think_start_id in ids
    assert tokenizer.think_end_id in ids
