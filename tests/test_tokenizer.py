import pytest

from shoujen.tokenizer import ShoujenTokenizer


def test_local_vocab_json_is_not_supported(tmp_path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="vocab.json"):
        ShoujenTokenizer.load(vocab_path)
