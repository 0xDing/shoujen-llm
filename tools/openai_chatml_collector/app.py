from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import gradio as gr
from dotenv import load_dotenv, set_key
from openai import OpenAI

from collector_core import (
    append_chatml_record,
    build_api_messages,
    build_completion_kwargs,
    build_single_turn_conversation,
    extract_assistant_content,
    normalize_conversation,
    parse_chatml_jsonl_output,
    parse_models,
    parse_prompt_lines,
    render_topic_prompt,
    resolve_dataset_path,
)


PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DATASET_PATH = "datasets/sft.jsonl"
SMART_TEMPLATE_PATH = PROJECT_DIR / "templates" / "smart_multiturn_chatml_zh.txt"


def load_settings() -> dict[str, str]:
    load_dotenv(ENV_FILE)
    load_dotenv()

    models_raw = os.getenv("MODELS", "")
    models = parse_models(models_raw)
    return {
        "base_url": os.getenv("BASE_URL", DEFAULT_BASE_URL),
        "api_key": os.getenv("API_KEY", ""),
        "models_raw": models_raw,
        "model": models[0] if models else "",
        "dataset_path": os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH),
    }


def load_smart_template() -> str:
    try:
        return SMART_TEMPLATE_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "编写5组chatml格式的多轮对话，用于训练文言文llm。\n本次生成的对话主题是：主题"


def chatbot_messages(conversation: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    return normalize_conversation(conversation)


def make_client(base_url: str, api_key: str) -> OpenAI:
    clean_api_key = (api_key or "").strip()
    if not clean_api_key:
        raise ValueError("API_KEY is empty")

    clean_base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
    return OpenAI(api_key=clean_api_key, base_url=clean_base_url)


def refresh_model_choices(models_raw: str, current_model: str | None) -> Any:
    choices = parse_models(models_raw)
    value = current_model if current_model in choices else (choices[0] if choices else current_model)
    return gr.update(choices=choices, value=value)


def save_config(base_url: str, api_key: str, models_raw: str) -> str:
    ENV_FILE.touch(mode=0o600, exist_ok=True)
    set_key(str(ENV_FILE), "BASE_URL", (base_url or "").strip())
    set_key(str(ENV_FILE), "API_KEY", (api_key or "").strip())
    set_key(str(ENV_FILE), "MODELS", ",".join(parse_models(models_raw)))
    return f"Saved {ENV_FILE}"


def clear_chat() -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    return [], [], "Cleared"


def undo_last_turn(
    conversation: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    messages = normalize_conversation(conversation)
    if not messages:
        return [], [], "Nothing to undo"

    while messages and messages[-1]["role"] != "user":
        messages.pop()
    if messages and messages[-1]["role"] == "user":
        messages.pop()

    return messages, chatbot_messages(messages), "Removed last turn"


def save_current_conversation(
    conversation: list[dict[str, Any]] | None,
    dataset_path: str,
    include_think: bool,
) -> str:
    output_path = resolve_dataset_path(dataset_path, PROJECT_DIR)
    saved_path = append_chatml_record(conversation, output_path, include_think=include_think)
    return f"Saved {saved_path}"


def add_manual_turn(
    user_message: str,
    assistant_message: str,
    conversation: list[dict[str, Any]] | None,
) -> tuple[Any, Any, list[dict[str, str]], list[dict[str, str]], str]:
    previous_messages = normalize_conversation(conversation)
    clean_user_message = (user_message or "").strip()
    clean_assistant_message = (assistant_message or "").strip()
    if not clean_user_message:
        return (
            gr.update(value=user_message),
            gr.update(value=assistant_message),
            chatbot_messages(previous_messages),
            previous_messages,
            "Manual user is empty",
        )
    if not clean_assistant_message:
        return (
            gr.update(value=user_message),
            gr.update(value=assistant_message),
            chatbot_messages(previous_messages),
            previous_messages,
            "Manual assistant is empty",
        )

    messages = [
        *previous_messages,
        {"role": "user", "content": clean_user_message},
        {"role": "assistant", "content": clean_assistant_message},
    ]
    return "", "", chatbot_messages(messages), messages, f"Added {len(messages) // 2} turn(s)"


def create_completion_from_conversation(
    conversation: list[dict[str, Any]],
    system_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    stop: str,
    frequency_penalty: float,
    presence_penalty: float,
    reasoning_effort: str,
) -> str:
    request_messages = build_api_messages(conversation, system_prompt)
    client = make_client(base_url, api_key)
    completion = client.chat.completions.create(
        **build_completion_kwargs(
            model=model.strip(),
            messages=request_messages,
            max_completion_tokens=int(max_completion_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stop=stop,
            frequency_penalty=float(frequency_penalty),
            presence_penalty=float(presence_penalty),
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
    )
    assistant_content = extract_assistant_content(completion.choices[0].message)
    if not assistant_content:
        raise ValueError("empty assistant response")
    return assistant_content


def create_single_turn_completion(
    prompt: str,
    system_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    stop: str,
    frequency_penalty: float,
    presence_penalty: float,
    reasoning_effort: str,
) -> str:
    return create_completion_from_conversation(
        [{"role": "user", "content": prompt}],
        system_prompt,
        base_url,
        api_key,
        model,
        max_completion_tokens,
        temperature,
        top_p,
        stop,
        frequency_penalty,
        presence_penalty,
        reasoning_effort,
    )


def send_message(
    user_message: str,
    conversation: list[dict[str, Any]] | None,
    system_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    stop: str,
    frequency_penalty: float,
    presence_penalty: float,
    reasoning_effort: str,
    auto_save: bool,
    include_think: bool,
    dataset_path: str,
) -> tuple[Any, list[dict[str, str]], list[dict[str, str]], str]:
    clean_user_message = (user_message or "").strip()
    previous_messages = normalize_conversation(conversation)
    if not clean_user_message:
        return gr.update(value=user_message), chatbot_messages(previous_messages), previous_messages, "Empty message"
    if not (model or "").strip():
        return gr.update(value=user_message), chatbot_messages(previous_messages), previous_messages, "Model is empty"

    messages = [*previous_messages, {"role": "user", "content": clean_user_message}]
    messages.append({"role": "assistant", "content": ""})

    try:
        assistant_content = create_completion_from_conversation(
            messages[:-1],
            system_prompt,
            base_url,
            api_key,
            model,
            int(max_completion_tokens),
            float(temperature),
            float(top_p),
            stop,
            float(frequency_penalty),
            float(presence_penalty),
            reasoning_effort,
        )

        messages[-1]["content"] = assistant_content
        status = "Done"
        if auto_save:
            output_path = resolve_dataset_path(dataset_path, PROJECT_DIR)
            saved_path = append_chatml_record(messages, output_path, include_think=include_think)
            status = f"Saved {saved_path}"
        return "", chatbot_messages(messages), messages, status
    except Exception as exc:
        return (
            gr.update(value=clean_user_message),
            chatbot_messages(previous_messages),
            previous_messages,
            f"Request failed: {exc}",
        )


def run_batch_prompts(
    prompts_text: str,
    system_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    stop: str,
    frequency_penalty: float,
    presence_penalty: float,
    reasoning_effort: str,
    concurrency: int,
    dataset_path: str,
    include_think: bool,
) -> Iterator[str]:
    prompts = parse_prompt_lines(prompts_text)
    if not prompts:
        yield "No prompts"
        return
    if not (model or "").strip():
        yield "Model is empty"
        return

    worker_count = max(1, int(concurrency or 4))
    output_path = resolve_dataset_path(dataset_path, PROJECT_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(prompts)
    successes = 0
    failures: list[str] = []
    yield f"Queued {total} prompts with concurrency={worker_count}"

    def infer_one(index: int, prompt: str) -> tuple[int, str, str]:
        assistant_content = create_single_turn_completion(
            prompt,
            system_prompt,
            base_url,
            api_key,
            model,
            int(max_completion_tokens),
            float(temperature),
            float(top_p),
            stop,
            float(frequency_penalty),
            float(presence_penalty),
            reasoning_effort,
        )
        return index, prompt, assistant_content

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(infer_one, index, prompt): (index, prompt)
            for index, prompt in enumerate(prompts, start=1)
        }
        for future in as_completed(futures):
            index, prompt = futures[future]
            try:
                _, prompt, assistant_content = future.result()
                conversation = build_single_turn_conversation(prompt, assistant_content)
                append_chatml_record(conversation, output_path, include_think=include_think)
                successes += 1
            except Exception as exc:
                failures.append(f"{index}: {exc}")

            completed = successes + len(failures)
            yield f"Completed {completed}/{total}; saved={successes}; failed={len(failures)}"

    status = f"Batch done. Saved {successes}/{total} rows to {output_path}"
    if failures:
        status += "\nFailures:\n" + "\n".join(failures[:20])
        if len(failures) > 20:
            status += f"\n... {len(failures) - 20} more"
    yield status


def run_smart_multiturn_batch(
    topics_text: str,
    template: str,
    system_prompt: str,
    base_url: str,
    api_key: str,
    model: str,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    stop: str,
    frequency_penalty: float,
    presence_penalty: float,
    reasoning_effort: str,
    concurrency: int,
    dataset_path: str,
    include_think: bool,
) -> Iterator[str]:
    topics = parse_prompt_lines(topics_text)
    if not topics:
        yield "No topics"
        return
    if not (template or "").strip():
        yield "Template is empty"
        return
    if not (model or "").strip():
        yield "Model is empty"
        return

    worker_count = max(1, int(concurrency or 4))
    output_path = resolve_dataset_path(dataset_path, PROJECT_DIR)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_topics = len(topics)
    completed_topics = 0
    saved_rows = 0
    ignored_items = 0
    failed_topics = 0
    failure_samples: list[str] = []
    yield f"Queued {total_topics} topics with concurrency={worker_count}"

    def infer_topic(index: int, topic: str) -> tuple[int, str, list[list[dict[str, str]]], list[str]]:
        prompt = render_topic_prompt(template, topic)
        assistant_content = create_single_turn_completion(
            prompt,
            system_prompt,
            base_url,
            api_key,
            model,
            int(max_completion_tokens),
            float(temperature),
            float(top_p),
            stop,
            float(frequency_penalty),
            float(presence_penalty),
            reasoning_effort,
        )
        conversations, errors = parse_chatml_jsonl_output(assistant_content, min_turns=2, max_turns=4)
        return index, topic, conversations, errors

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(infer_topic, index, topic): (index, topic)
            for index, topic in enumerate(topics, start=1)
        }
        for future in as_completed(futures):
            index, topic = futures[future]
            completed_topics += 1
            try:
                _, _, conversations, errors = future.result()
                ignored_items += len(errors)
                for conversation in conversations:
                    try:
                        append_chatml_record(conversation, output_path, include_think=include_think)
                        saved_rows += 1
                    except Exception as exc:
                        ignored_items += 1
                        if len(failure_samples) < 20:
                            failure_samples.append(f"{index}/{topic}: save ignored: {exc}")
                if errors and len(failure_samples) < 20:
                    failure_samples.append(f"{index}/{topic}: {errors[0]}")
            except Exception as exc:
                failed_topics += 1
                if len(failure_samples) < 20:
                    failure_samples.append(f"{index}/{topic}: request failed: {exc}")

            yield (
                f"Completed {completed_topics}/{total_topics} topics; "
                f"saved_rows={saved_rows}; ignored={ignored_items}; failed_topics={failed_topics}"
            )

    status = (
        f"Smart batch done. Saved {saved_rows} rows to {output_path}; "
        f"ignored={ignored_items}; failed_topics={failed_topics}"
    )
    if failure_samples:
        status += "\nSamples:\n" + "\n".join(failure_samples)
    yield status


def build_ui() -> gr.Blocks:
    settings = load_settings()
    models = parse_models(settings["models_raw"])
    smart_template_value = load_smart_template()

    with gr.Blocks(title="OpenAI ChatML Collector") as demo:
        conversation_state = gr.State([])
        manual_conversation_state = gr.State([])

        gr.Markdown("## OpenAI ChatML Collector")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Chat"):
                        chatbot = gr.Chatbot(height=560, label="Conversation")
                        user_input = gr.Textbox(lines=4, label="User", placeholder="输入消息后发送")
                        with gr.Row():
                            send_button = gr.Button("Send", variant="primary")
                            undo_button = gr.Button("Undo")
                            clear_button = gr.Button("Clear")
                        with gr.Row():
                            auto_save = gr.Checkbox(False, label="Auto save")
                            save_button = gr.Button("Save")
                        status = gr.Textbox(label="Status", interactive=False)

                    with gr.Tab("Batch"):
                        batch_prompts = gr.Textbox(
                            lines=18,
                            label="Prompts",
                            placeholder="每行一个 prompt；每行都会作为单轮 user 消息请求并保存",
                        )
                        with gr.Row():
                            concurrency = gr.Number(value=4, precision=0, label="Concurrency")
                            batch_button = gr.Button("Run batch", variant="primary")
                        batch_status = gr.Textbox(lines=10, label="Batch status", interactive=False)

                    with gr.Tab("Smart Batch"):
                        smart_topics = gr.Textbox(lines=10, label="Topics", placeholder="每行一个主题")
                        with gr.Row():
                            smart_concurrency = gr.Number(value=4, precision=0, label="Concurrency")
                            smart_button = gr.Button("Run smart batch", variant="primary")
                        smart_status = gr.Textbox(lines=10, label="Smart batch status", interactive=False)
                        with gr.Accordion("Template", open=False):
                            smart_template = gr.Textbox(
                                value=smart_template_value,
                                lines=20,
                                label="Fixed template",
                                interactive=False,
                            )

                    with gr.Tab("Manual"):
                        manual_chatbot = gr.Chatbot(height=420, label="Manual conversation")
                        manual_user = gr.Textbox(lines=4, label="User")
                        manual_assistant = gr.Textbox(lines=8, label="Assistant")
                        with gr.Row():
                            manual_add_button = gr.Button("Add turn", variant="primary")
                            manual_undo_button = gr.Button("Undo")
                            manual_clear_button = gr.Button("Clear")
                            manual_save_button = gr.Button("Save manual")
                        manual_status = gr.Textbox(label="Manual status", interactive=False)

            with gr.Column(scale=1):
                system_prompt = gr.Textbox(lines=6, label="System prompt")
                base_url = gr.Textbox(settings["base_url"], label="BASE_URL")
                api_key = gr.Textbox(settings["api_key"], type="password", label="API_KEY")
                models_raw = gr.Textbox(settings["models_raw"], lines=3, label="MODELS")
                model = gr.Dropdown(
                    choices=models,
                    value=settings["model"],
                    label="Model",
                    allow_custom_value=True,
                )
                dataset_path = gr.Textbox(settings["dataset_path"], label="Dataset path")
                include_think = gr.Checkbox(True, label="Include <think>")
                with gr.Row():
                    refresh_button = gr.Button("Refresh")
                    save_config_button = gr.Button("Save config")
                max_completion_tokens = gr.Number(value=2048, precision=0, label="Max completion tokens")
                temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Temperature")
                top_p = gr.Slider(0.0, 1.0, value=0.95, step=0.01, label="Top P")
                stop = gr.Textbox(lines=2, label="Stop", placeholder="留空为 None，多行表示多个 stop")
                frequency_penalty = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Frequency penalty")
                presence_penalty = gr.Slider(-2.0, 2.0, value=0.0, step=0.05, label="Presence penalty")
                reasoning_effort = gr.Dropdown(
                    choices=["", "none", "minimal", "low", "medium", "high", "xhigh"],
                    value="",
                    label="Reasoning effort (OpenRouter)",
                )

        send_inputs = [
            user_input,
            conversation_state,
            system_prompt,
            base_url,
            api_key,
            model,
            max_completion_tokens,
            temperature,
            top_p,
            stop,
            frequency_penalty,
            presence_penalty,
            reasoning_effort,
            auto_save,
            include_think,
            dataset_path,
        ]
        send_outputs = [user_input, chatbot, conversation_state, status]

        user_input.submit(send_message, inputs=send_inputs, outputs=send_outputs)
        send_button.click(send_message, inputs=send_inputs, outputs=send_outputs)
        save_button.click(save_current_conversation, inputs=[conversation_state, dataset_path, include_think], outputs=status)
        clear_button.click(clear_chat, outputs=[conversation_state, chatbot, status])
        undo_button.click(undo_last_turn, inputs=conversation_state, outputs=[conversation_state, chatbot, status])
        batch_button.click(
            run_batch_prompts,
            inputs=[
                batch_prompts,
                system_prompt,
                base_url,
                api_key,
                model,
                max_completion_tokens,
                temperature,
                top_p,
                stop,
                frequency_penalty,
                presence_penalty,
                reasoning_effort,
                concurrency,
                dataset_path,
                include_think,
            ],
            outputs=batch_status,
        )
        smart_button.click(
            run_smart_multiturn_batch,
            inputs=[
                smart_topics,
                smart_template,
                system_prompt,
                base_url,
                api_key,
                model,
                max_completion_tokens,
                temperature,
                top_p,
                stop,
                frequency_penalty,
                presence_penalty,
                reasoning_effort,
                smart_concurrency,
                dataset_path,
                include_think,
            ],
            outputs=smart_status,
        )
        manual_add_button.click(
            add_manual_turn,
            inputs=[manual_user, manual_assistant, manual_conversation_state],
            outputs=[manual_user, manual_assistant, manual_chatbot, manual_conversation_state, manual_status],
        )
        manual_save_button.click(
            save_current_conversation,
            inputs=[manual_conversation_state, dataset_path, include_think],
            outputs=manual_status,
        )
        manual_clear_button.click(
            clear_chat,
            outputs=[manual_conversation_state, manual_chatbot, manual_status],
        )
        manual_undo_button.click(
            undo_last_turn,
            inputs=manual_conversation_state,
            outputs=[manual_conversation_state, manual_chatbot, manual_status],
        )
        refresh_button.click(refresh_model_choices, inputs=[models_raw, model], outputs=model)
        models_raw.change(refresh_model_choices, inputs=[models_raw, model], outputs=model)
        save_config_button.click(save_config, inputs=[base_url, api_key, models_raw], outputs=status)

    return demo


if __name__ == "__main__":
    build_ui().launch()
