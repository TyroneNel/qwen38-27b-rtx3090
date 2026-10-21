#!/usr/bin/env python3
"""X-16 / H16 probe: import every in-repo CUDA-extension package with no GPU.

Each case runs in a fresh subprocess (module-level side effects must not leak
between cases) under one of two modes:
  bare  - vLLM absent (the true state of this box).
  stub  - permissive in-process `vllm.*` stub (meta-path finder below), so a
          remaining failure is NOT "vLLM is missing". KVarN's own files are
          loaded as vllm.* modules from kvarn/files (install.sh copies them
          there in the image); triton is the real triton 3.7.1 (CPU wheel).
Per case the child reports OK | FALLBACK (no exception, but the feature is
off) | FAIL (exception type, message, raise site). Raise site = innermost
traceback frame; `repo_frame` = innermost frame inside the worktree.

  python probe.py [WORKTREE]          # default ~/research-wt/x16
"""
import json, os, subprocess, sys, traceback
from pathlib import Path

WT = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else
          os.path.expanduser("~/research-wt/x16")).resolve()

# ---------------------------------------------------------------- cases
# (id, mode, env, code). `code` runs in the child; it may set `fallback = "why"`.
FA2_BACKEND = "import fa2_fp8kv_vllm.backend as b"
_pl = (WT/"patches/spec-decode-attn.patch").read_text().splitlines()
_i = next(i for i,l in enumerate(_pl) if l.startswith("+++ b/v1/attention/ops/spec_decode_attn.py"))
_body = [l[1:] for l in _pl[_i+2:] if l.startswith("+")]
_f = Path(os.environ.get("TMPDIR","/tmp"))/"x16-stub"; _f.mkdir(exist_ok=True)
(_f/"spec_decode_attn.py").write_text("\n".join(_body)+"\n")
SPEC_SRC = ("import importlib.util as u\nsp=u.spec_from_file_location('vllm.v1.attention.ops.spec_decode_attn','%s')\n"
            "m=u.module_from_spec(sp); sp.loader.exec_module(m)" % (_f/"spec_decode_attn.py"))
CASES = [
 # --- pthfa (pth_flash_attn) ---
 ("P1 pth_flash_attn",                "bare", {}, "import pth_flash_attn"),
 ("P2 pth_flash_attn.planner",        "bare", {}, "import pth_flash_attn.planner"),
 ("P3 pth_flash_attn.ops",            "bare", {}, "from pth_flash_attn import ops\nif not ops.is_available(): fallback='ops.is_available() False (no _C.abi3.so), import succeeds'"),
 ("P3b ops.pth_flash_attn_fwd() call","bare", {}, "from pth_flash_attn import ops\nops.pth_flash_attn_fwd(*([None]*15))"),
 ("P4 plugin import",                 "bare", {}, "import pth_flash_attn.plugin"),
 ("P4b plugin.register()",            "bare", {}, "import pth_flash_attn.plugin as p; p.register()"),
 ("P4c plugin.register()",            "stub", {}, "import pth_flash_attn.plugin as p; p.register()"),
 ("P5 vllm_backend import",           "bare", {}, "import pth_flash_attn.vllm_backend"),
 ("P5s vllm_backend import",          "stub", {}, "import pth_flash_attn.vllm_backend"),
 ("P5t supports_compute_capability(8,6)", "stub", {},
  "import pth_flash_attn.vllm_backend as v\nfrom vllm.platforms.interface import DeviceCapability\n"
  "ok = v.PthFlashAttnBackend.supports_compute_capability(DeviceCapability(8,6))\n"
  "if not ok: fallback='backend reports sm86 unsupported (ops.is_available() False); vLLM then rejects/skips CUSTOM'"),
 # --- fa2 (fa2_fp8kv_vllm) ---
 ("F1 fa2_fp8kv_vllm",                "bare", {}, "import fa2_fp8kv_vllm"),
 ("F2 paths.check()",                 "bare", {}, "import fa2_fp8kv_vllm.paths as p; p.check()"),
 ("F3 register() FA2_FP8KV unset",    "bare", {}, "import fa2_fp8kv_vllm as f; f.register()"),
 ("F4 register() FA2_FP8KV=1",        "bare", {"FA2_FP8KV":"1"}, "import fa2_fp8kv_vllm as f; f.register()"),
 ("F4s register() FA2_FP8KV=1",       "stub", {"FA2_FP8KV":"1"}, "import fa2_fp8kv_vllm as f; f.register()"),
 ("F5 backend import",                "bare", {}, FA2_BACKEND),
 ("F6 fa2_prefill import",            "bare", {}, "import fa2_prefill"),
 ("F7 paged_prefill import",          "bare", {}, "import paged_prefill"),
 ("F5a backend, CPU (cc None)",       "stub", {"STUB_CC":"none"}, FA2_BACKEND),
 ("F5b backend, sm80 A100, FA2=1",    "stub", {"STUB_CC":"8.0","FA2_FP8KV":"1"}, FA2_BACKEND+"\nfallback='no .so load on sm80; extension_selected() False -> native FLASH_ATTN'"),
 ("F5c backend, sm86, FA2=1, .so missing", "stub", {"STUB_CC":"8.6","FA2_FP8KV":"1"}, FA2_BACKEND),
 ("F5e backend, sm120, FA2=1",        "stub", {"STUB_CC":"12.0","FA2_FP8KV":"1"}, FA2_BACKEND+"\nfallback='no .so load on sm120; extension_selected() False -> native FLASH_ATTN'"),
 ("F6s fa2_prefill import",           "stub", {}, "import fa2_prefill"),
 # --- patch-created Triton module (first-version text reconstructed from the + lines of the patch) ---
 ("S1 spec_decode_attn.py (patch 1 text)", "stub", {}, SPEC_SRC),
 # --- kvarn (installed INTO vllm.*; standalone not importable) ---
 ("K1 kvarn sinkhorn (torch only)",   "stub", {}, "import vllm.model_executor.layers.quantization.kvarn.sinkhorn"),
 ("K2 kvarn_decode (torch only)",     "stub", {}, "import vllm.v1.attention.ops.kvarn_decode"),
 ("K3 kvarn config",                  "stub", {}, "import vllm.model_executor.layers.quantization.kvarn.config"),
 ("K4 kvarn_store",                   "stub", {}, "import vllm.v1.attention.ops.kvarn_store"),
 ("K5 triton_kvarn_sinkhorn",         "stub", {}, "import vllm.v1.attention.ops.triton_kvarn_sinkhorn"),
 ("K6 triton_kvarn_decode",           "stub", {}, "import vllm.v1.attention.ops.triton_kvarn_decode"),
 ("K7 kvarn_attn backend",            "stub", {}, "import vllm.v1.attention.backends.kvarn_attn"),
 ("K0 kvarn_attn backend",            "bare", {}, "import vllm.v1.attention.backends.kvarn_attn"),
 ("K8 kvarn Triton sinkhorn run on CPU tensor", "stub", {},
  "import torch\nfrom vllm.v1.attention.ops.triton_kvarn_sinkhorn import kvarn_sinkhorn_triton\n"
  "kvarn_sinkhorn_triton(torch.randn(2,128,128), iterations=2)"),
]

# ---------------------------------------------------------------- stub
STUB = r'''
import importlib.abc, importlib.util, os, sys, types
from pathlib import Path
_WT = Path(os.environ["X16_WT"])
KV = _WT / "kvarn/files"
class _M(type):
    def __getattr__(c, n):
        if n.startswith("__"): raise AttributeError(n)
        return _mk(n)
    def __call__(c, *a, **k):
        if len(a) == 1 and not k and callable(a[0]) and not isinstance(a[0], _M): return a[0]
        return type.__call__(c, *a, **k)
    def __getitem__(c, i): return c
    def __or__(c, o): return c
    __ror__ = __or__
class _Any:
    def __init__(self, *a, **k): pass
def _mk(n): return _M(n, (_Any,), {"__module__": "vllm_stub"})
class _Mod(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith("__"): raise AttributeError(n)
        v = _mk(n); setattr(self, n, v); return v
class _Cap:
    def __init__(s, major, minor=0): s.major, s.minor = major, minor
    def to_int(s): return s.major*10+s.minor
class _Plat:
    def get_device_capability(self):
        cc = os.environ.get("STUB_CC", "none")
        return None if cc == "none" else tuple(int(x) for x in cc.split("."))
    def __getattr__(self, n): return _mk(n)
class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != "vllm" and not name.startswith("vllm."): return None
        real = None
        if name != "vllm":
            rel = name.split(".", 1)[1].replace(".", "/")
            for cand in (KV/"vllm"/(rel + ".py"), KV/"vllm"/rel/"__init__.py"):
                if cand.is_file(): real = cand
        if real:
            return importlib.util.spec_from_file_location(name, real,
                submodule_search_locations=[str(real.parent)] if real.name == "__init__.py" else None)
        return importlib.util.spec_from_loader(name, self, is_package=True)
    def create_module(self, spec): return _Mod(spec.name)
    def exec_module(self, m):
        m.__path__ = []
        n = m.__name__
        if n == "vllm.envs": m.__getattr__ = lambda k: None  # unset env var -> None
        if n == "vllm.platforms": m.current_platform = _Plat()
        if n == "vllm.platforms.interface": m.DeviceCapability = _Cap
        if n == "vllm.triton_utils":
            import triton, triton.language as tl; m.triton, m.tl = triton, tl
        if n == "vllm.v1.attention.backends.fa_utils":
            m.is_flash_attn_varlen_func_available = lambda: False
sys.meta_path.insert(0, _Finder())
'''

CHILD = r'''
import os, sys, json, traceback
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
WT = os.environ["X16_WT"]
sys.path[:0] = [WT + "/pthfa/python", WT + "/fa2/python"]
mode, code = sys.argv[1], sys.argv[2]
if mode == "stub": exec(open(os.environ["X16_STUBFILE"]).read())
ns = {"fallback": None}
res = {}
try:
    exec(code, ns)
    res = {"status": "FALLBACK" if ns.get("fallback") else "OK", "note": ns.get("fallback")}
except BaseException as e:
    tb = traceback.extract_tb(e.__traceback__)
    # frames that are code from the worktree (our package/kvarn files), not the exec'd snippet
    inwt = [f for f in tb if f.filename.startswith(WT)]
    def fm(f): return f"{os.path.relpath(f.filename, WT)}:{f.lineno}"
    res = {"status": "FAIL", "exc": type(e).__name__, "msg": str(e)[:300],
           "repo_frame": fm(inwt[-1]) if inwt else None,
           "raise_site": f"{f.filename.replace(os.path.expanduser('~'), '~')}:{f.lineno}" if (f := tb[-1]) else None,
           "repo_line": inwt[-1].line if inwt else None}
print("@@" + json.dumps(res))
'''

def main():
    sd = Path(os.environ.get("TMPDIR", "/tmp")) / "x16-stub"
    sd.mkdir(exist_ok=True)
    stubfile = sd / "vllm_stub.py"; stubfile.write_text(STUB)
    print(f"worktree: {WT}  python: {sys.version.split()[0]}  torch: ", end="")
    print(subprocess.run([sys.executable, "-c", "import torch;print(torch.__version__, 'cuda', torch.cuda.is_available())"],
                         capture_output=True, text=True).stdout.strip())
    print(f"{'case':44} {'mode':5} {'status':9} detail")
    rows = []
    for cid, mode, env, code in CASES:
        e = dict(os.environ, X16_WT=str(WT), X16_STUBFILE=str(stubfile), **env)
        e.pop("FA2_FP8KV", None) if "FA2_FP8KV" not in env else None
        e.pop("PYTHONPATH", None)
        p = subprocess.run([sys.executable, "-c", CHILD, mode, code], env=e, capture_output=True, text=True, cwd="/tmp")
        line = [l for l in p.stdout.splitlines() if l.startswith("@@")]
        r = json.loads(line[-1][2:]) if line else {"status": "CRASH", "msg": p.stderr[-300:]}
        rows.append((cid, mode, r))
        if r["status"] == "FAIL":
            d = f"{r['exc']}: {r['msg']!r} @ repo {r['repo_frame']}  [{r['repo_line']}]  site {r['raise_site']}"
        else:
            d = r.get("note") or r.get("msg") or ""
        print(f"{cid:44} {mode:5} {r['status']:9} {d}")
    print("\nSUMMARY", {s: sum(1 for _,_,r in rows if r['status']==s) for s in ("OK","FALLBACK","FAIL","CRASH")})

main()
