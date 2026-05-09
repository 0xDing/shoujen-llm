from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector_core import (  # noqa: E402
    append_chatml_record,
    build_api_messages,
    build_chatml_record,
    build_completion_kwargs,
    build_openrouter_extra_body,
    build_single_turn_conversation,
    extract_assistant_content,
    is_openrouter_base_url,
    normalize_reasoning_effort,
    parse_chatml_jsonl_output,
    parse_models,
    parse_prompt_lines,
    parse_stop,
    prepare_conversation_for_storage,
    remove_markdown_fences,
    render_topic_prompt,
    resolve_dataset_path,
    strip_think_blocks,
    validate_strict_chatml_record,
)


class CollectorCoreTest(unittest.TestCase):
    def test_parse_models_deduplicates_and_strips(self) -> None:
        self.assertEqual(parse_models(" a, b,\na, ,c "), ["a", "b", "c"])

    def test_parse_prompt_lines_strips_empty_lines(self) -> None:
        self.assertEqual(parse_prompt_lines(" a \n\n b\n  "), ["a", "b"])

    def test_render_topic_prompt_replaces_fixed_topic_marker(self) -> None:
        template = "编写数据\n本次生成的对话主题是：主题"
        self.assertEqual(render_topic_prompt(template, "茶道"), "编写数据\n本次生成的对话主题是：茶道")

    def test_render_topic_prompt_supports_braced_placeholder(self) -> None:
        self.assertEqual(render_topic_prompt("topic={topic}", "琴谱"), "topic=琴谱")

    def test_api_messages_include_system_prompt(self) -> None:
        messages = build_api_messages(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
            "be concise",
        )

        self.assertEqual(messages[0], {"role": "system", "content": "be concise"})
        self.assertEqual(messages[1:], [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])

    def test_parse_stop(self) -> None:
        self.assertIsNone(parse_stop(""))
        self.assertEqual(parse_stop("END"), "END")
        self.assertEqual(parse_stop("END\nSTOP"), ["END", "STOP"])

    def test_remove_markdown_fences(self) -> None:
        self.assertEqual(remove_markdown_fences("```jsonl\n{}\n```"), "{}")

    def test_completion_kwargs_match_openai_sdk_shape(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        kwargs = build_completion_kwargs(
            model="mimo-v2.5-pro",
            messages=messages,
            max_completion_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
        )

        self.assertEqual(kwargs["model"], "mimo-v2.5-pro")
        self.assertEqual(kwargs["messages"], messages)
        self.assertEqual(kwargs["max_completion_tokens"], 1024)
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertEqual(kwargs["top_p"], 0.95)
        self.assertFalse(kwargs["stream"])
        self.assertIsNone(kwargs["stop"])
        self.assertEqual(kwargs["frequency_penalty"], 0.0)
        self.assertEqual(kwargs["presence_penalty"], 0.0)

    def test_openrouter_reasoning_extra_body(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        kwargs = build_completion_kwargs(
            model="openai/o3-mini",
            messages=messages,
            max_completion_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
            base_url="https://openrouter.ai/api/v1",
            reasoning_effort="high",
        )

        self.assertEqual(kwargs["extra_body"], {"reasoning": {"effort": "high"}})

    def test_openrouter_reasoning_extra_body_is_conditional(self) -> None:
        self.assertTrue(is_openrouter_base_url("https://openrouter.ai/api/v1"))
        self.assertFalse(is_openrouter_base_url("https://api.openai.com/v1"))
        self.assertEqual(normalize_reasoning_effort(" XHIGH "), "xhigh")
        self.assertEqual(normalize_reasoning_effort("default"), "")
        self.assertEqual(build_openrouter_extra_body("https://openrouter.ai/api/v1", ""), None)
        self.assertEqual(build_openrouter_extra_body("https://api.openai.com/v1", "high"), None)

    def test_validate_strict_chatml_record_accepts_2_to_4_turns(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ]
        }
        self.assertEqual(validate_strict_chatml_record(record), record["messages"])

    def test_validate_strict_chatml_record_rejects_bad_roles(self) -> None:
        with self.assertRaises(ValueError):
            validate_strict_chatml_record(
                {"messages": [{"role": "system", "content": "x"}, {"role": "assistant", "content": "a"}]}
            )

    def test_parse_chatml_jsonl_output_ignores_invalid_records(self) -> None:
        valid = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "u1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "u2"},
                    {"role": "assistant", "content": "a2"},
                ]
            },
            ensure_ascii=False,
        )
        conversations, errors = parse_chatml_jsonl_output(f"```jsonl\nnot json\n{valid}\n```")
        self.assertEqual(len(conversations), 1)
        self.assertTrue(errors)

    def test_parse_chatml_jsonl_output_accepts_json_array(self) -> None:
        record = {
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ]
        }
        conversations, errors = parse_chatml_jsonl_output(json.dumps([record], ensure_ascii=False))
        self.assertEqual(conversations, [record["messages"]])
        self.assertEqual(errors, [])

    def test_chatml_record_excludes_system_and_preserves_think_tags(self) -> None:
        record = build_chatml_record(
            [
                {"role": "system", "content": "hidden"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<think>reason</think>\nanswer"},
            ]
        )
        self.assertEqual(
            record,
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "<think>reason</think>\nanswer"},
                ]
            },
        )

    def test_chatml_record_can_filter_think_tags(self) -> None:
        record = build_chatml_record(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "<think>reason</think>\nanswer"},
            ],
            include_think=False,
        )

        self.assertEqual(
            record,
            {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]},
        )

    def test_strip_think_blocks_handles_multiple_blocks(self) -> None:
        self.assertEqual(strip_think_blocks("<think>a</think>\nanswer\n<think>b</think>"), "answer")

    def test_prepare_conversation_for_storage_only_filters_assistant(self) -> None:
        messages = prepare_conversation_for_storage(
            [
                {"role": "user", "content": "<think>keep</think> question"},
                {"role": "assistant", "content": "<think>drop</think> answer"},
            ],
            include_think=False,
        )
        self.assertEqual(messages[0]["content"], "<think>keep</think> question")
        self.assertEqual(messages[1]["content"], "answer")

    def test_build_single_turn_conversation(self) -> None:
        self.assertEqual(
            build_single_turn_conversation("prompt", "answer"),
            [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    def test_extract_assistant_content_wraps_reasoning_content(self) -> None:
        message = SimpleNamespace(content="answer", reasoning_content="reason")
        self.assertEqual(extract_assistant_content(message), "<think>reason</think>\nanswer")

    def test_extract_assistant_content_does_not_duplicate_existing_think_tags(self) -> None:
        content = "<think>reason</think>\nanswer"
        message = SimpleNamespace(content=content, reasoning_content="another reason")
        self.assertEqual(extract_assistant_content(message), content)

    def test_append_chatml_record_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "data" / "sft.jsonl"
            append_chatml_record(
                [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好。"},
                ],
                path,
            )

            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(row["messages"][0]["content"], "你好")

    def test_resolve_dataset_path_uses_project_base_for_relative_paths(self) -> None:
        base_dir = Path("/tmp/project")
        self.assertEqual(resolve_dataset_path("datasets/a.jsonl", base_dir), base_dir / "datasets/a.jsonl")


if __name__ == "__main__":
    unittest.main()
