# camoufler

**What it does.** Camoufler takes what you typed — slang, shorthand, an unfinished ask — and turns it into clear standard English. It fills in missing pieces of the request and strips personal details (emails, phone numbers, passwords, account numbers) before anything leaves your machine.

**Why.** So you can talk the way you talk, send a complete request, and keep your identity and PII out of it. You stay unique. Your digital presence stays hidden.

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
