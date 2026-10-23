#!/usr/bin/env python3
"""attn-4 offline study, GPU step (docs/research/deepinfer/attention-kv.md "attn-4: training-free top-k sparse
attention"): capture the real post-RoPE Q and K of the target's 16 full-attention layers on one ~128k cold prefill
plus greedy decode steps, for bench/attn4_mass.py.

vLLM runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0, enforce_eager, as bench/act_calib.py) with mode D's KV
setup: int8_per_token_head on the pthfa CUSTOM backend (--backend TRITON_ATTN as a fallback), no drafter, prefix
caching off, 2,048-token prefill chunks. A forward pre-hook on every target `layers.N.self_attn.attn` (vLLM's
Attention module; it receives q and k after q/k-norm and RoPE, exactly what the cache and the kernel see) records:
  K   every position, bf16 bits -> L<NN>.k.npy  [positions, Hkv, D] uint16 (memmap; ~256 MB per layer at 128k)
  Q   the last --q-tail prompt positions, every decode position, and --q-spread positions spread over the
      prompt (one query every ctx/q-spread tokens, the depth curve) -> L<NN>.q.npy [nq, Hq, D] uint16, and
      qpos.npy [nq] (absolute positions; one request, so the order of the hook calls is the position order)
K is captured before the int8 cache write: the selection study is about which keys matter, not about the int8
grid (the int8 error is attn-5's subject). meta.json has the shapes, the attention scale, the prompt length and the
tokens actually seen per layer (= prompt + generated - 1, checked). The prompt: the repo's docs/**/*.md
concatenated in path order (no repetition), cut to --ctx tokens, then a question; chat template, thinking off.

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python bench/attn4_dump.py --model /app/models/Qwen3.8-27B-W4A16-AutoRound-fast \
      --out /cache/di-dumps/attn4 [--ctx 128000 --gen 64 --q-tail 32 --q-spread 16]
  python bench/attn4_dump.py --prompt-only ...    CPU: build and tokenize the prompt, print its length, exit
Exit 0 = dumped and consistent, 2 = inconsistent (positions seen != expected), 1 = error.
"""
import argparse
import glob
import json
import os
import re
import sys

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--model", default="/app/models/Qwen3.8-27B-W4A16-AutoRound-fast")
ap.add_argument("--out", required=True)
ap.add_argument("--ctx", type=int, default=128000, help="document tokens in the prompt")
ap.add_argument("--gen", type=int, default=64, help="greedy decode tokens")
ap.add_argument("--q-tail", type=int, default=32)
ap.add_argument("--q-spread", type=int, default=16)
ap.add_argument("--docs", default=None, help="docs root (default: <repo>/docs, else /repo/docs)")
ap.add_argument("--backend", default="CUSTOM", help="attention backend (CUSTOM = pthfa; TRITON_ATTN fallback)")
ap.add_argument("--gpu-util", type=float, default=0.92)
ap.add_argument("--prompt-only", action="store_true")
a = ap.parse_args()

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
QUESTION = ("Using only the documentation above, list the five most important measured results it reports, "
            "with their numbers, and say which document each comes from.")


def docs_text(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))
    parts = []
    for p in files:
        with open(p, encoding="utf-8", errors="replace") as f:
            parts.append(f"## {os.path.relpath(p, root)}\n\n{f.read()}")
    return "\n\n".join(parts)


def build_prompt(tok):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = a.docs or next(r for r in (os.path.join(here, "docs"), "/repo/docs") if os.path.isdir(r))
    ids = tok.encode(docs_text(root), add_special_tokens=False)
    if len(ids) < a.ctx:
        sys.exit(f"attn4_dump: the docs corpus has {len(ids)} tokens < --ctx {a.ctx}")
    doc = tok.decode(ids[:a.ctx])
    msgs = [{"role": "user", "content": "Documentation:\n\n" + doc + "\n\n" + QUESTION}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


if a.prompt_only:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    p = build_prompt(tok)
    print(f"ATTN4_PROMPT tokens={len(p)} (doc {a.ctx}) + gen {a.gen} -> max_model_len needed {len(p) + a.gen}")
    sys.exit(0)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.forward_context import get_forward_context  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

os.makedirs(a.out, exist_ok=True)
llm = LLM(model=a.model, served_model_name="q", gpu_memory_utilization=a.gpu_util, max_model_len=a.ctx + 2048,
          max_num_seqs=1, enforce_eager=True, language_model_only=True, kv_cache_dtype="int8_per_token_head",
          attention_backend=a.backend, mamba_ssm_cache_dtype="float16", max_num_batched_tokens=2048,
          enable_prefix_caching=False, async_scheduling=False)
tok = llm.get_tokenizer()
prompt = build_prompt(tok)
n_prompt = len(prompt)
total = n_prompt + a.gen
step = max(1, n_prompt // max(a.q_spread, 1))
q_spread = set(range(step - 1, n_prompt - a.q_tail, step)) if a.q_spread else set()


def find_model(obj, depth=0, seen=None):
    seen = seen if seen is not None else set()
    if depth > 14 or id(obj) in seen:
        return None
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        return obj if any(n.endswith("self_attn.attn") for n, _ in obj.named_modules()) else None
    for attr in ("llm_engine", "engine_core", "model_executor", "driver_worker", "worker", "model_runner", "model",
                 "engine", "_model", "runner"):
        if hasattr(obj, attr):
            try:
                r = find_model(getattr(obj, attr), depth + 1, seen)
            except Exception:
                continue
            if r is not None:
                return r
    return None


model = find_model(llm)
assert model is not None, "could not find the torch model"
armed = False
state = {}


def make_hook(li, mod):
    st = state[li] = {"seen": 0, "q": [], "qpos": [], "k": None, "hq": mod.num_heads, "hkv": mod.num_kv_heads,
                      "d": mod.head_size, "scale": float(mod.impl.scale)}

    def hook(m, args):
        if not armed:
            return
        q, k = args[0], args[1]
        d = st["d"]
        q = q.reshape(q.shape[0], -1, d)
        k = k.reshape(k.shape[0], -1, d)
        t = q.shape[0]
        # A padded batch (none expected under enforce_eager) carries num_actual_tokens in its metadata.
        md = get_forward_context().attn_metadata
        if isinstance(md, dict):
            md = md.get(m.layer_name)
        if md is None:
            return  # a dummy / profile forward
        t = min(t, int(getattr(md, "num_actual_tokens", t) or t))
        q, k = q[:t], k[:t]
        p0 = st["seen"]
        if st["k"] is None:
            st["k"] = np.lib.format.open_memmap(os.path.join(a.out, f"L{li:02d}.k.npy"), mode="w+", dtype=np.uint16,
                                                shape=(total, k.shape[1], d))
        if p0 + t > total:
            raise RuntimeError(f"layer {li}: {p0 + t} positions > expected {total}")
        st["k"][p0:p0 + t] = k.to(torch.bfloat16).view(torch.int16).cpu().numpy().view(np.uint16)
        pos = torch.arange(p0, p0 + t)
        sel = [i for i, pp in enumerate(pos.tolist()) if pp >= n_prompt - a.q_tail or pp in q_spread]
        if sel:
            st["q"].append(q[torch.tensor(sel, device=q.device)].to(torch.bfloat16).view(torch.int16).cpu().numpy())
            st["qpos"] += [p0 + i for i in sel]
        st["seen"] = p0 + t
    return hook


layers = []
for name, mod in model.named_modules():
    mi = re.search(r"layers\.(\d+)\.self_attn\.attn$", name)
    if mi and "mtp" not in name and type(mod).__name__ == "Attention":
        li = int(mi.group(1))
        mod.register_forward_pre_hook(make_hook(li, mod))
        layers.append(li)
print(f"attn4_dump: hooks on {len(layers)} layers {layers}; prompt {n_prompt} tokens, gen {a.gen}", flush=True)
assert layers, "no full-attention layer found"
armed = True
outs = llm.generate([TokensPrompt(prompt_token_ids=prompt)], SamplingParams(max_tokens=a.gen, temperature=0,
                                                                             ignore_eos=True))
armed = False
gen_ids = list(outs[0].outputs[0].token_ids)
expect = n_prompt + len(gen_ids) - 1
meta = {"model": a.model, "backend": a.backend, "prompt_tokens": n_prompt, "gen_tokens": len(gen_ids),
        "expected_positions": expect, "q_tail": a.q_tail, "q_spread": sorted(q_spread), "layers": {}}
bad = []
for li in layers:
    st = state[li]
    st["k"].flush()
    qa = np.concatenate(st["q"]) if st["q"] else np.zeros((0, st["hq"], st["d"]), np.uint16)
    np.save(os.path.join(a.out, f"L{li:02d}.q.npy"), qa.view(np.uint16))
    np.save(os.path.join(a.out, f"L{li:02d}.qpos.npy"), np.array(st["qpos"], dtype=np.int64))
    meta["layers"][f"{li:02d}"] = {"seen": st["seen"], "nq": int(qa.shape[0]), "hq": st["hq"], "hkv": st["hkv"],
                                   "d": st["d"], "scale": st["scale"]}
    if st["seen"] != expect:
        bad.append((li, st["seen"]))
meta["answer"] = tok.decode(gen_ids)
with open(os.path.join(a.out, "meta.json"), "w") as f:
    json.dump(meta, f, indent=1)
print(f"attn4_dump: answer: {meta['answer'][:300]!r}", flush=True)
print(f"ATTN4_DUMP layers={len(layers)} prompt={n_prompt} gen={len(gen_ids)} positions={expect} "
      f"nq={meta['layers'][f'{layers[0]:02d}']['nq']} inconsistent={bad}", flush=True)
sys.exit(2 if bad else 0)
