# OpenAI ChatML Collector

Gradio WebUI for collecting multi-turn SFT conversations through an OpenAI-compatible Chat Completions API.

## Setup

```bash
cd tools/openai_chatml_collector
cp .env.example .env
uv run python app.py
```

`.env` keys:

```dotenv
BASE_URL=https://api.openai.com/v1
API_KEY=
MODELS=your-model-1,your-model-2
DATASET_PATH=datasets/sft.jsonl
```

`MODELS` is comma-separated and can also be edited in the WebUI.

## Dataset Format

Each saved conversation appends one JSONL row:

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"<think>...</think>\n你好。"}]}
```

The WebUI sends the configured system prompt to the API request, but saved JSONL rows only keep `user` and `assistant` messages. Assistant output is saved verbatim, so `<think>...</think>` stays inside the assistant message content.

## API Call

The app calls the official OpenAI Python SDK directly:

```python
client = OpenAI(api_key=api_key, base_url=base_url)
completion = client.chat.completions.create(
    model=model,
    messages=messages,
    max_completion_tokens=max_completion_tokens,
    temperature=temperature,
    top_p=top_p,
    stream=False,
    stop=stop,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
)
```

When `BASE_URL` points to `https://openrouter.ai/api/v1` and `Reasoning effort (OpenRouter)` is set, the app also sends:

```python
extra_body={"reasoning": {"effort": reasoning_effort}}
```

Supported effort values are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`. Leaving the WebUI field blank omits `extra_body`.

`Include <think>` controls dataset output. It is checked by default. When unchecked, assistant `<think>...</think>` blocks are removed before writing JSONL.

## Controls

- `Send`: call the selected model with the current system prompt and conversation history.
- `Save`: append the current conversation to `DATASET_PATH`.
- `Auto save`: append the conversation after each successful assistant response.
- `Undo`: remove the last user/assistant turn from the current conversation.
- `Batch`: paste one prompt per line, run single-turn calls concurrently, and append one JSONL row per successful prompt. Default concurrency is `4`.
- `Smart Batch`: paste one topic per line. The app renders the fixed template, asks the model to generate 5 multi-turn ChatML JSONL records per topic, validates each record, saves valid rows, and ignores invalid rows.
- `Manual`: fill `User` and `Assistant` yourself, add one or more turns, then save the hand-written conversation without calling an API.
- `Include <think>`: keep assistant thinking blocks by default; uncheck it to filter them from saved rows.
- `Save config`: write `BASE_URL`, `API_KEY`, and `MODELS` to local `.env`.
