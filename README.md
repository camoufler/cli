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

## Running

On a terminal, run `camoufler` with no flags. A menu offers:

1. **download** — pull a small CPU model (prompt defaults to the saved model)
2. **list** — local and remote **1.5B** Ollama tags (current default is marked)
3. **set** — save the default 1.5B model used by standardize (`~/.camoufler/settings.json`)
4. **standardize** — rewrite/expand interactively (Enter to send; Ctrl+C or Ctrl+D to return to the menu)

Until you **set** a model, standardize uses `qwen2.5:1.5b`. Prompt config defaults to `config/config.example2.json` when that file is present.

Piped (flags still work):

```text
echo "gonna head out later" | camoufler -f standardize
```

PowerShell:

```text
"gonna head out later" | camoufler -f standardize
```

Pull a model without the menu:

```text
camoufler -f download -m qwen2.5:1.5b
```

| Flag | Description |
|------|-------------|
| `--model` / `-m` | Ollama `name:tag`; overrides the saved default |
| `--function` / `-f` | `download`, `list`, `set`, or `standardize` (omit on a TTY for the menu) |
| `--config` / `-c` | JSON config; if omitted, example2 then example then the packaged default |
| `--version` | Print version |
| `--verbose` | `0` errors, `1` info, `2` debug |

GPU offload is disabled (`num_gpu: 0`).
