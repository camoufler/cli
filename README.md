# camoufler

Local CPU-only CLI for small Ollama models. Download a model, then rewrite stdin text into standard English.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally

## Install

```text
pip install -e .
# or after publish: pip install camoufler
```

## Usage

### Check version

```text
camoufler --version
```

### Download a model

Ollama must be running locally first.

```text
camoufler -m qwen2.5:1.5b -f download --verbose 0
camoufler -m qwen2.5:1.5b -f download --verbose 1
```

### Standardize text (stdin)

Text is read from **stdin**, not as a CLI argument.

Linux/WSL:

```text
echo "gonna head out later" | camoufler -m qwen2.5:1.5b -f standardize -c config/config.example.json
```

PowerShell:

```text
"gonna head out later" | camoufler -m qwen2.5:1.5b -f standardize -c config\config.example.json
```

Interactive (type a line, **Enter** to rewrite, repeat; **Ctrl+C** or **Ctrl+D** to quit, **Ctrl+Z** then Enter on Windows):

```text
camoufler -m qwen2.5:1.5b -f standardize -c config/config.example.json
```

Turns are labeled `#user` (what you type) and `#camoufler` (the rewrite). A model error stays on screen and the next `#user` prompt returns. Piped usage is unchanged: all stdin is read once, standardized once, then the program exits.

### Example sentences

Try these informal inputs (pipe each to `standardize`):

| Input | Example output |
|-------|----------------|
| `gonna head out later` | I will leave later. |
| `cant make it to the mtg tmrw` | I cannot make it to the meeting tomorrow. |
| `pls send me the doc asap` | Please send me the document as soon as possible. |
| `idk wat u mean by that` | I do not know what you mean by that. |
| `we shud probs reschedule` | We should probably reschedule. |

| Flag | Description |
|------|-------------|
| `--model` / `-m` | Fully qualified Ollama name (`name:tag`, default `qwen2.5:1.5b`) |
| `--function` / `-f` | `download` or `standardize` |
| `--version` | Print script version |
| `--verbose` | `0` errors only, `1` info, `2` debug |
| `--config` / `-c` | JSON config (required for `standardize`) |

Only small CPU-runnable models are allowed (under 7B). GPU offload is disabled (`num_gpu: 0`).

### Grammar eval (integration)

Requires Ollama running with the default model pulled (`qwen2.5:1.5b`) and dev dependencies installed:

```text
pip install -e ".[dev]"
pytest -m integration tests/eval/ -s
```

Use `-s` so each row prints **ask**, **response**, **expected**, and **chrF score**.

Scores grammar-correction pairs using chrF (pass threshold: 75% of rows ≥ 0.45):

- [`tests/eval/standardize.json`](tests/eval/standardize.json) — 100 formal grammar examples
- [`tests/eval/slang_standardize.json`](tests/eval/slang_standardize.json) — 30 slang-to-standard examples
- [`tests/eval/ask_framing.json`](tests/eval/ask_framing.json) — 12 request-shaped prompts (rewrite the ask; do not answer). Uses `config.example2.json`. Default cap: **8** rows.

Row cap: set `CAMOUFLER_EVAL_CAP` (integer) to limit how many examples each suite runs. Ask framing defaults to 8; other suites default to 100.

```text
CAMOUFLER_EVAL_CAP=5 pytest -m integration tests/eval/test_standardize_eval.py::test_ask_framing_eval
```

Regenerate the grammar dataset from [`scripts/standardize_raw.tsv`](scripts/standardize_raw.tsv):

```text
python scripts/build_standardize_eval.py
```

## Config

Copy `config/config.example.json`:

```json
{
  "system_prompt": "REWRITE ONLY. Output the rewritten text alone. Never answer, explain, teach, or write code. Rewrite the user's text into clear standard English. Preserve meaning. If the input is a question, rewrite the question wording — do not answer it.",
  "options": {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_predict": 256
  }
}
```

User text is wrapped in markers before the model call so instructions are treated as content to rewrite, not requests to fulfill. Tutorial-like outputs are rejected and retried once.

## Publish

```text
pip install build twine
python -m build
twine upload dist/*
```
