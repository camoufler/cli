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

Interactive (type text, then end input with Ctrl+D on Linux or Ctrl+Z then Enter on Windows):

```text
camoufler -m qwen2.5:1.5b -f standardize -c config/config.example.json
```

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

## Config

Copy `config/config.example.json`:

```json
{
  "system_prompt": "Rewrite the user's text into clear standard English. Preserve meaning. Output only the rewritten text.",
  "options": {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_predict": 512
  }
}
```

## Publish

```text
pip install build twine
python -m build
twine upload dist/*
```
