#!/usr/bin/env bash
# X-17 CPU checks (no GPU). Writes nothing outside a temp dir; stdout is checks.out.
#   bash checks.sh > checks.out
# Baseline = commit 20c7625 of the scratch repo ~/research-wt/vllm-0.31.0-timing
# (pristine v0.31.0 + patches/series + the four KVarN patches; series-apply.out).
set -u; export PYTHONDONTWRITEBYTECODE=1
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WT=${WT:-$HOME/research-wt/vllm-0.31.0-timing}
REPO=$(cd -- "$HERE/../../../../.." && pwd)
P=$HERE/spec-timing.patch
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git -C "$WT" archive 20c7625 vllm | tar -x -C "$T"
cd "$T/vllm"
echo "# X-17 checks, $(date -u +%FT%TZ), baseline $(git -C "$WT" log -1 --format='%h %s' 20c7625)"

echo; echo "## 1. apply check against the baseline tree"
git apply --check -p1 "$P" && echo "git apply --check: OK"
patch -p1 --fuzz 0 --dry-run -i "$P"; echo "patch --fuzz 0 --dry-run exit=$?"

echo; echo "## 1b. the stock image tree: patches/ and kvarn/ at 2cf50a0 (image commit) vs this branch"
git -C "$REPO" diff --quiet 2cf50a0 -- patches/ kvarn/ && echo "patches/ + kvarn/ identical to 2cf50a0: same tree as the baseline"

echo; echo "## 2. real apply (Dockerfile.variant flags), then compare with the scratch tree HEAD"
patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -i "$P"; echo "apply exit=$?"
for f in v1/worker/gpu/model_runner.py v1/worker/gpu/spec_timing.py; do
  git -C "$WT" show "HEAD:vllm/$f" | cmp -s - "$f" && echo "identical to scratch HEAD $(git -C "$WT" log -1 --format=%h): $f"
done

echo; echo "## 3. second apply must fail (--forward)"
patch -p1 --forward --fuzz 0 --dry-run -i "$P" >/dev/null; echo "second apply exit=$? (nonzero expected)"

echo; echo "## 3b. verify.sh patch checks (entrypoint runs verify.sh --no-server on every boot): baseline vs variant"
chk() { for n in $(bash "$REPO/patches/apply.sh" --list); do
          if patch -p1 -R --dry-run -s --fuzz 0 -d "$1" < "$REPO/patches/$n" >/dev/null 2>&1; then r=rev
          elif python3 "$REPO/patches/_check_applied.py" "$REPO/patches/$n" "$1" >/dev/null 2>&1; then r=content; else r=FAIL; fi
          echo "$n $r"; done
        for n in $(bash "$REPO/patches/apply.sh" --list --kvarn); do
          patch -p1 -R --dry-run -s --fuzz 0 -d "$1" < "$REPO/kvarn/$n" >/dev/null 2>&1 && echo "$n rev" || echo "$n FAIL"; done; }
mkdir "$T/b"; git -C "$WT" archive 20c7625 vllm | tar -x -C "$T/b"
chk "$T/b/vllm" > "$T/v0"; chk "$T/vllm" > "$T/v1"
diff "$T/v0" "$T/v1" && echo "same result on all $(wc -l < "$T/v1") patches ($(grep -c ' rev$' "$T/v1") exact reverse, $(grep -c ' content$' "$T/v1") content check, $(grep -c ' FAIL$' "$T/v1") FAIL = $(grep ' FAIL$' "$T/v1" | cut -d' ' -f1 | tr '\n' ' ')which verify.sh passes as superseded)"

echo; echo "## 4. byte-compile, no .pyc written ($(python3 --version); the image uses 3.12)"
for f in v1/worker/gpu/model_runner.py v1/worker/gpu/spec_timing.py "$HERE/parse_timing.py" "$HERE/test_timing.py"; do
  python3 -c 'import sys; compile(open(sys.argv[1]).read(), sys.argv[1], "exec")' "$f" && echo "compile OK: ${f#$HERE/}"
done

echo; echo "## 5. unit tests (fake event times; module extracted from spec-timing.patch)"
(cd "$T" && python3 "$HERE/test_timing.py"); echo "exit=$?"

echo; echo "## 6. parser on a SYNTHETIC log (fake events, NOT measurements)"
python3 - "$HERE" "$T/syn.log" <<'PY'
import sys, torch
sys.argv = [sys.argv[0]] + sys.argv[1:]
here, out = sys.argv[1], sys.argv[2]
src = open(f"{here}/test_timing.py").read().split("# 1. phases")[0]
ns = {"__file__": f"{here}/test_timing.py"}; exec(compile(src, "test_timing.py", "exec"), ns)
dev, lines, t = ns["mk"](every=5)
x = ns["step"](dev, t, 0.0, (20, 1.2, 0.4, 0.3, 0), drafts=0, graph=False, draft=False, emitted=(1,))
for i in range(9):
    dev.done = x
    x = ns["step"](dev, t, x + 1.2, (14, 1.2, 1.8, 0.6, 3.9), emitted=(4,))
dev.done = 1e9; t.end(); t.emit()
open(out, "w").write("INFO noise\n" + "\n".join(f"(EngineCore pid=7) INFO {l}" for l in lines) + "\n")
PY
python3 "$HERE/parse_timing.py" "$T/syn.log" --label synthetic
