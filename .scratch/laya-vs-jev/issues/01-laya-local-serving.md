# Laya: install and serve locally on Apple Silicon

Type: research
Status: resolved
Blocked by:

## Question

How do we install and run Laya on an Apple M5 (16 GB, macOS) so it answers decisions through a Jev-compatible HTTP API? Pin down: the Python package name and version, the Hugging Face checkpoint(s) and their sizes, the serving command (official server, Railway template's image, or MLX port), whether it runs on MPS/MLX or CPU only, and the request/response shape compared to Jev's (endpoint paths, question types such as Choice / Score / Noul, how confidence is returned). Can one client talk to both?

## Answer

Findings: branch `research/laya-local-serving`, file `research/laya-local-serving.md` (commit 9de1ec8). Nothing installed yet.

- Upstream route (chosen default): `uv pip install "laya[serve]"` (PyPI `laya` 0.4.17), run `LAYA_HOST=127.0.0.1 LAYA_DEVICE=mps laya-serve` → `POST /v1/systemone`, `POST /v1/systemone/batch`, `GET /health` on :8000. PyTorch, fp32 on MPS, one request at a time. Always set `LAYA_HOST` (defaults to 0.0.0.0).
- Alternatives: community `laya-apple` 1.7.0 (MLX + ANE, :8642, has `/v1/models`, wire format of upstream 0.3.20); `laya-mlx` 0.6.0 (source of the 13.4 ms M3 Max figure).
- Checkpoints (Apache-2.0): `convaiinnovations/laya` (ModernBERT-large 421M, ~843 MB, 512 tokens), `convaiinnovations/laya-multilingual` (322M), `convaiinnovations/laya-typed-decisions` (421M). ~2 GB RAM each loaded fp32.
- Wire: request identical to Jev; response is a superset (`routing`, `answer_confidence`, `action`, `x_jev_confidence`); `LAYA_JEV_STRICT=1` returns exactly Jev's contract.
- Confidence differs: Laya `confidence` = 1 − normalised entropy; Jev = (n·p_max−1)/(n−1). For a fair comparison, compute confidence ourselves from `probabilities`, or read `x_jev_confidence`.
- One client works: `typesafe_sdk` / `lados.py` / `jev.py` just need `TYPESAFE_BASE_URL` → local server. Upstream lacks `/v1/models`; choices >100 options need `LAYA_TOURNAMENT_GROUP` (irrelevant for our ≤7-option fields).
- Open: no M5 benchmarks; real latency to be measured in ticket 07.
