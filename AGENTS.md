# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Running tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies.
# requirements/test/cuda.txt is pinned to x86_64; on other platforms, use the
# unpinned source file instead:
uv pip install -r requirements/test/cuda.in    # resolves for current platform
# Or on x86_64:
uv pip install -r requirements/test/cuda.txt

# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

---

## Cursor Cloud specific instructions

This VM has **no GPU**, so vLLM is built and run as its **CPU backend**
(`VLLM_TARGET_DEVICE=cpu`; see `docs/getting_started/installation/cpu.md`). The
snapshot already contains a ready `.venv` with an editable, CPU-compiled vLLM.

- **Use the prebuilt venv.** Run everything via `.venv/bin/python` /
  `.venv/bin/vllm` (or `source .venv/bin/activate`). `uv` lives in `~/.local/bin`
  (`astral.sh` is blocked, so it was installed from PyPI, not the install script).
- **Egress is restricted; `huggingface.co` is blocked.** Models cannot be pulled
  from the Hub. For runs and tests, point at a **local model directory** and set
  `HF_HUB_OFFLINE=1`; use `--load-format dummy` to skip weight downloads (random
  weights, so output is gibberish but the full engine path is exercised). Many
  `-m cpu_test` tests still fetch model configs from the Hub and will fail —
  prefer HF-independent tests (e.g. `tests/test_outputs.py`, `tests/test_inputs.py`).
- **C/C++ rebuilds must force gcc.** `/usr/bin/c++` is clang and fails to link
  libstdc++; always build with `CC=gcc CXX=g++`. Python-only edits need **no**
  rebuild (editable install). C++/CMake/kernel edits require a manual rebuild —
  the startup update script does not rebuild:
  `CC=gcc CXX=g++ VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation --index-strategy unsafe-best-match`.
  (`--index-strategy unsafe-best-match` is required whenever installing the
  `requirements/*.txt` because they add the pytorch CPU extra index.)
- **Running on CPU.** Set `VLLM_CPU_KVCACHE_SPACE` (GiB) and
  `VLLM_CPU_OMP_THREADS_BIND` (e.g. `0-6`, reserve a core for the front-end), and
  pass `--dtype bfloat16 --enforce-eager`. `vllm serve <local_dir> ... --port 8000`
  exposes the OpenAI API; a from-scratch tokenizer has no chat template, so pass
  `--chat-template` for `/v1/chat/completions`.
