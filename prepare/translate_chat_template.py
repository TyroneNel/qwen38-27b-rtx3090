"""Translate a Qwen3.8 chat template's reasoning-effort vocabulary so client
supplied effort levels can no longer 400 the server.

The shipped chat_template.jinja accepts exactly xhigh/medium/low and defaults
to xhigh. OpenAI-protocol clients speak the gpt-5 vocabulary —
none/minimal/low/medium/high/xhigh/max — so `minimal` (and `high`, `max`)
raise jinja2.TemplateError inside the template and vLLM surfaces that as a
400 Bad Request on every request that carries one (docs/gotchas.md 58).
This rewrites the effort block in place to:

  - default to `medium` (the repo's serving default; was xhigh),
  - alias the OpenAI vocabulary onto the template's three levels
    (minimal -> low, high/max -> xhigh; none reaches the template only with
    thinking off, where the block is skipped anyway),
  - fall back to `medium` for anything unrecognized instead of raising.

Idempotent: a `{#- effort-translation v1 ... -#}` marker short-circuits
re-runs, so prepare.sh can call this on every boot, and a re-download that
clobbers chat_template.jinja is re-translated on the next prepare. Refuses to
touch a template whose effort block does not match the expected original
shape (upstream changed) — never blind-edits, and exits nonzero so prepare
fails loudly instead of silently leaving the 400 live.

Usage: python prepare/translate_chat_template.py MODEL_DIR [MODEL_DIR ...]
"""

import shutil
import sys
from pathlib import Path

MARKER = "{#- effort-translation v1: managed by prepare/translate_chat_template.py"

# The exact original shape (chat_template.jinja lines 47-50): the default line,
# the vocabulary guard, the raise, and its endif. Matched as a structural span
# (first line located by anchor, next three by shape), not as one brittle blob.
ANCHOR = "set resolved_reasoning_effort = reasoning_effort|default("
SHAPES = (
    "if resolved_reasoning_effort not in",
    "raise_exception(",
    "endif",
)

REPLACEMENT = """\
    {#- effort-translation v1: managed by prepare/translate_chat_template.py.
        OpenAI/gpt-5 vocabulary -> template levels; default medium; unknown
        values fall back to medium instead of raising (docs/gotchas.md 58). -#}
    {%- set effort_aliases = {'minimal': 'low', 'low': 'low', 'medium': 'medium', 'high': 'xhigh', 'xhigh': 'xhigh', 'max': 'xhigh'} %}
    {%- set requested_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 'medium' %}
    {%- set resolved_reasoning_effort = effort_aliases.get(requested_effort, 'medium') %}"""


def translate(path: Path) -> str:
    """Rewrite one chat_template.jinja. Returns translated|already|missing|foreign."""
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already"
    lines = text.splitlines(keepends=True)
    idx = next((i for i, l in enumerate(lines) if ANCHOR in l), None)
    if idx is None:
        return "foreign"
    span = lines[idx + 1 : idx + 1 + len(SHAPES)]
    if not all(shape in span[i] for i, shape in enumerate(SHAPES)):
        return "foreign"
    shutil.copy2(path, path.with_suffix(".jinja.bak-effort"))
    lines[idx : idx + 1 + len(SHAPES)] = [REPLACEMENT + "\n"]
    path.write_text("".join(lines), encoding="utf-8")
    return "translated"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: translate_chat_template.py MODEL_DIR [MODEL_DIR ...]")
    failed = []
    for arg in sys.argv[1:]:
        path = Path(arg.rstrip("/")) / "chat_template.jinja"
        try:
            result = translate(path)
        except OSError as exc:
            result = f"write failed ({exc.strerror})"
        print(f"translate_chat_template: {path}: {result}")
        if result not in ("translated", "already", "missing"):
            failed.append(f"{path}: {result}")
    if failed:
        sys.exit(
            "translate_chat_template: left unchanged —\n  "
            + "\n  ".join(failed)
            + "\n(unknown effort block means the template changed upstream; "
            + "refusing to blind-edit. Fix permissions or re-check the shape.)"
        )


if __name__ == "__main__":
    main()
