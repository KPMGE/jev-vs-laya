# Setup

How to get Laya running locally and talk to it (and to Jev) through one client.

## Requirements

- macOS on Apple Silicon (developed on an M5, 16 GB). Linux/CUDA works too: set `LAYA_DEVICE=cuda`.
- [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and every dependency.
- ~3 GB free disk: PyTorch plus the 843 MB English Laya checkpoint (cached by Hugging Face under `~/.cache/huggingface`).

## Install

```bash
cd laya-vs-jev
uv sync
cp .env.example .env        # fill keys later; everything runs without them
```

`uv sync` installs `laya[serve]==0.4.17` (Convai Innovations, Apache-2.0), `httpx` and `python-dotenv`.

## Keys

| Variable | Needed for | Without it |
| --- | --- | --- |
| `TYPESAFE_API_KEY` | Calling Jev (`console.typesafe.ai`) | Jev is skipped everywhere |
| `ANTHROPIC_API_KEY` | Labeling the dataset with Claude | Labeling script refuses to run |

Optional overrides: `LAYA_BASE_URL` (default `http://127.0.0.1:8000`), `TYPESAFE_BASE_URL` (default `https://api.typesafe.ai`).

## Serve Laya

```bash
./scripts/serve_laya.sh
```

This runs `laya-serve` with:

| Env | Value | Why |
| --- | --- | --- |
| `LAYA_HOST` | `127.0.0.1` | `laya-serve` binds `0.0.0.0` by default; keep it off the network |
| `LAYA_PORT` | `8000` | |
| `LAYA_DEVICE` | `mps` | Mac GPU, fp32 |
| `LAYA_MODELS` / `LAYA_DEFAULT_MODEL` | `english` | `convaiinnovations/laya`, ModernBERT-large, 421M params, 512-token states |

The first start downloads the checkpoint from Hugging Face. Check it is up with `curl http://127.0.0.1:8000/health`. It reports the device it actually uses (a failed GPU request falls back to CPU silently).

`laya-serve` handles one request at a time; concurrent requests queue.

## Smoke test

In a second terminal:

```bash
uv run scripts/smoke.py
```

It sends one support message with one `choice` question (`issue_type`) to Laya, and to Jev if `TYPESAFE_API_KEY` is set, and prints answer, confidence, latency and input tokens. The first call per backend is a discarded warm-up.

## The client

`src/triage/client.py` is the only code that talks to a model. A backend is a base URL, a key and a model name:

```python
from triage.client import DecisionClient, laya, jev

with DecisionClient(laya()) as c:
    d = c.decide("Message text", {"issue_type": {...}})
d.answers["issue_type"]["choice"], d.confidence("issue_type"), d.latency_s
```

- Laya and Jev take the same request, `POST /v1/systemone` with `{model, state, questions}`. Question types are `choice`, `score` and `noul`.
- **Confidence is recomputed from `probabilities` with Jev's formula** `(n·p_max − 1)/(n − 1)`. Laya's own `confidence` field uses 1 − normalised entropy, so the raw fields aren't comparable. `noul` answers carry only P(true), so they have no confidence.
- `jev()` returns `None` when `TYPESAFE_API_KEY` is unset. Jev is pinned to `jev-1.14.0` and priced at $0.042 per million input tokens (output is free).

## Measured on this machine

Apple M5, 16 GB, macOS, `laya` 0.4.17, PyTorch 2.15, `LAYA_DEVICE=mps`, measured 2026-10-29:

| Check | Result |
| --- | --- |
| `/health` | `device: mps`, checkpoint `english` at revision `7b928d82…`, 0 CPU fallbacks |
| Smoke message ("charged twice … or I'm cancelling") | `billing`, P = 0.997, Jev-formula confidence 0.997 |
| Latency, one 7-option choice, 123 input tokens, 20 warm calls | p50 87 ms, p95 98 ms |

The response's `model` field reads `laya-rl-agent` (the server's routed name), not `english`.
