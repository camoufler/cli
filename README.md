# camoufler

Local CPU-only CLI for small Ollama models.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running locally

## Setup

```text
pip install -e .
```

Pull a small CPU model (`name:tag`, under 7B):

```text
camoufler -m qwen2.5:1.5b -f download
```

## Running

Piped:

```text
echo "gonna head out later" | camoufler -m qwen2.5:1.5b -f standardize -c config/config.example.json
```

PowerShell:

```text
"gonna head out later" | camoufler -m qwen2.5:1.5b -f standardize -c config\config.example.json
```

Interactive (Enter to send; Ctrl+C or Ctrl+D to quit):

```text
camoufler -m qwen2.5:1.5b -f standardize -c config/config.example.json
```

| Flag | Description |
|------|-------------|
| `--model` / `-m` | Ollama name (`name:tag`, default `qwen2.5:1.5b`) |
| `--function` / `-f` | `download` or `standardize` |
| `--config` / `-c` | JSON config (required for `standardize`) |
| `--version` | Print version |
| `--verbose` | `0` errors, `1` info, `2` debug |

GPU offload is disabled (`num_gpu: 0`).

## Usecase modes

`standardize` classifies stdin, rewrites it to standard English, then redacts PII.

| Mode | What it does |
|------|----------------|
| Utterance | Slang or grammar with no ask → one rewritten line |
| Prompt | Real ask → one expanded paragraph (does not fulfill the ask) |

Prompt types:

| Type | Framework |
|------|-----------|
| Factual | RTF |
| Instructional | TAG |
| Creative | CREATE |
| Analytical | RACE |
| Transformation | TRAC |
| Role-playing | COAST |
| Strategic | GRADE |
