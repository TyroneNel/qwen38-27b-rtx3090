"""Capture GPTQ Hessians for the DFlash2 drafter's linear layers from the drafter's OWN
inputs on real traffic: vLLM in-process (no engine multiprocessing), eager mode (CUDA
graphs/torch.compile would bypass module hooks), the bf16 drafter doing speculative
decoding on our self-distillation prompts (drafter/data/gen.jsonl: same mix as the draft
vocabulary and the MTP Hessians), model-default sampling.

Hooks on the fused vLLM modules: layers.N.self_attn.qkv_proj (input 5120), o_proj (4096),
mlp.gate_up_proj (5120), mlp.down_proj (17408), and fc (25600). Inputs with K <= 5120 are
accumulated into fp32 Hessians on the GPU; the wide ones (down_proj, fc) are dumped as bf16
rows to memmaps and reduced on the GPU after the engine is torn down (a 17408^2 fp32
Hessian is 1.2 GB; five of them plus fc's 2.6 GB don't fit next to the model).

  DRAFT=models/Qwen3.8-27B-DFlash2 python drafter/capture_dflash2.py [--prompts 400] [--max-tokens 384] [--rows 250000]
  -> drafter/runs/dflash2/hessians.pt  {module_key: {"H": fp32 [K,K], "n": rows}}

--dump-target OUT (drafter-2): a different job, the input of drafter/eval_dflash2.py. The
target alone (no drafter) generates a reference response per prompt (greedy and/or the
model-default sampling), then one prefill pass over prompt+response per sequence dumps the
target rows the DFlash2 drafter consumes: the hidden states after the layers in the
drafter's target_layer_ids (hidden + residual, the aux-hidden-state value vLLM concatenates
into fc) for the response and the last window-1 prompt rows, plus the target's top-64
raw logits and logsumexp per row. Layout: drafter/dflash2_ref.py (Capture).

  MODEL=models/Qwen3.8-27B-W4A16-AutoRound-fast python drafter/capture_dflash2.py --dump-target OUT
      [--domains real_think,real_nothink,code,danish,chat,think,gsm8k_think,gsm8k_nothink,labd]
      [--n 12] [--skip 0] [--prompt-seed 0] [--modes greedy,sample] [--seeds 0]
      [--max-tokens 768] [--labd-ctx 16000] [--labd-corpus ~/bench/labd_corpus.txt]
      [--gsm8k DIR] [--draft-config DIR]
"""
import os, sys, json, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
sys.path.insert(0, HERE)
from gptq_utils import accumulate_hessian

TARGET = os.environ.get("MODEL", os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound"))
DRAFT = os.environ.get("DRAFT", os.path.join(REPO, "models", "Qwen3.8-27B-DFlash2"))
OUT = os.environ.get("OUT", os.path.join(HERE, "runs", "dflash2"))
K_SPEC = int(os.environ.get("K", 7))
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
NPROMPTS, MAXTOK, ROWCAP = arg("--prompts", 400), arg("--max-tokens", 384), arg("--rows", 250000)
SMALL_K = 5120          # accumulate on GPU up to this input width
HOOKED = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj", "fc")
os.makedirs(OUT, exist_ok=True)


def find_draft_model(root):
    """BFS over attributes for an nn.Module whose class name contains 'DFlash'."""
    seen, queue = set(), [(root, "runner", 0)]
    while queue:
        obj, path, depth = queue.pop(0)
        if id(obj) in seen or depth > 4:
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module) and "DFlash" in type(obj).__name__ and hasattr(obj, "model"):
            return obj, path
        if isinstance(obj, torch.nn.Module):
            continue
        d = getattr(obj, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if k.startswith("__"):
                continue
            queue.append((v, f"{path}.{k}", depth + 1))
    return None, None


def main():
    from vllm import LLM, SamplingParams
    recs = [json.loads(l) for l in open(os.path.join(HERE, "data", "gen.jsonl"))][:NPROMPTS]
    print(f"{len(recs)} prompts, max_tokens {MAXTOK}, k={K_SPEC}", flush=True)

    llm = LLM(
        model=TARGET, served_model_name="qwen3.8-27b",
        speculative_config={"method": "dflash", "model": DRAFT, "num_speculative_tokens": K_SPEC},
        enforce_eager=True,
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.95)),
        max_model_len=int(os.environ.get("MAX_LEN", 4096)), max_num_seqs=int(os.environ.get("MAX_SEQS", 8)),
        max_num_batched_tokens=2048, attention_backend="FLASH_ATTN", kv_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="float16", language_model_only=True, enable_prefix_caching=False,
    )
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    runner = core.model_executor.driver_worker.worker.model_runner
    draft, path = find_draft_model(runner)
    assert draft is not None, "could not find the DFlash draft model under the model runner"
    print("draft model at", path, type(draft).__name__, flush=True)

    # ---- hooks -------------------------------------------------------------------
    stats = {}   # key -> dict(H, n) on gpu | dict(mm, n, K) memmap
    handles = []
    for name, mod in draft.named_modules():
        key = next((h for h in HOOKED if name.endswith(h)), None)
        if key is None or not isinstance(mod, torch.nn.Module) or not hasattr(mod, "forward"):
            continue
        if key == "fc" and not (name == "fc" or name.endswith(".fc")):
            continue
        mkey = name[len("model."):] if name.startswith("model.") else name
        K = getattr(mod, "input_size", None) or getattr(mod, "input_size_per_partition", None)
        if K is None:
            continue
        if K <= SMALL_K:
            stats[mkey] = {"H": torch.zeros(K, K, device="cuda", dtype=torch.float32), "n": 0, "K": K}
        else:
            mm = np.lib.format.open_memmap(os.path.join(OUT, f"rows_{mkey}.npy"), mode="w+",
                                           dtype=np.uint16, shape=(ROWCAP, K))
            stats[mkey] = {"mm": mm, "n": 0, "K": K}

        def pre_hook(m, args, mkey=mkey):
            x = args[0]
            X = x.reshape(-1, x.shape[-1])
            st = stats[mkey]
            if "H" in st:
                st["H"], st["n"] = accumulate_hessian(st["H"], X, st["n"])
            else:
                n = st["n"]
                if n < ROWCAP:
                    take = min(X.shape[0], ROWCAP - n)
                    st["mm"][n:n + take] = X[:take].to(torch.bfloat16).cpu().view(torch.uint16).numpy()
                    st["n"] = n + take
        handles.append(mod.register_forward_pre_hook(pre_hook))
    # The k/v rows of qkv_proj are ALSO applied to the context rows, through the fused
    # context-KV precompute (_project_context_kv: rms_norm(context_states, hidden_norm)
    # @ W_kv) which bypasses qkv_proj.forward. This captures that input distribution as
    # "ctx_kv"; quant_dflash2.py blends it into the k/v Hessians when present. Measured
    # 7% WORSE greedy acceptance than calibrating k/v on the query rows alone, so the
    # shipped drafter is built from a Hessian file with this key removed (drafter/README.md).
    import torch.nn.functional as F
    from vllm import _custom_ops as ops
    dm = draft.model
    stats["ctx_kv"] = {"H": torch.zeros(dm.fc.output_size if hasattr(dm.fc, "output_size") else 5120,
                                        dm.fc.output_size if hasattr(dm.fc, "output_size") else 5120,
                                        device="cuda", dtype=torch.float32), "n": 0, "K": 5120}
    orig_pck = dm._project_context_kv

    def _project_context_kv(context_states, *a, **kw):
        normed = torch.empty_like(context_states)
        ops.rms_norm(normed, context_states, dm._hidden_norm_weight, dm._rms_norm_eps)
        st = stats["ctx_kv"]
        st["H"], st["n"] = accumulate_hessian(st["H"], normed.reshape(-1, normed.shape[-1]), st["n"])
        return orig_pck(context_states, *a, **kw)
    dm._project_context_kv = _project_context_kv
    print(f"hooked {len(stats)} modules: {sorted(stats)}", flush=True)
    assert len(stats) == 4 * 5 + 2, f"expected 22 hooked modules, got {len(stats)}"

    sp = SamplingParams(max_tokens=MAXTOK, temperature=1.0, top_p=0.95, top_k=20)
    t0 = time.time()
    CH = 32
    for i in range(0, len(recs), CH):
        batch = recs[i:i + CH]
        outs = llm.generate([{"prompt_token_ids": r["prompt_ids"]} for r in batch], sp, use_tqdm=False)
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        n_small = stats["layers.0.self_attn.qkv_proj"]["n"]
        n_wide = stats["layers.0.mlp.down_proj"]["n"]
        print(f"[{i + len(batch)}/{len(recs)}] {ntok} new tokens; rows small={n_small} wide={n_wide} "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    for h in handles:
        h.remove()
    dm._project_context_kv = orig_pck
    res = {}
    for k, st in stats.items():
        if "H" in st:
            res[k] = {"H": st["H"].cpu(), "n": st["n"]}
        else:
            st["mm"].flush()
    torch.save(res, os.path.join(OUT, "hessians_small.pt"))
    json.dump({k: {"n": st["n"], "K": st["K"]} for k, st in stats.items() if "H" not in st},
              open(os.path.join(OUT, "rows_meta.json"), "w"))
    # The in-process engine does not give its GPU memory back on del; re-exec into the
    # reduce phase with a clean process instead.
    print("capture done; re-executing for the Hessian reduction", flush=True)
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), "--reduce-only"])


def reduce_wide():
    """Phase 2: X^T X for the dumped wide inputs on the (now free) GPU."""
    res = torch.load(os.path.join(OUT, "hessians_small.pt"))
    meta = json.load(open(os.path.join(OUT, "rows_meta.json")))
    for k, m in meta.items():
        n, K = m["n"], m["K"]
        mm = np.load(os.path.join(OUT, f"rows_{k}.npy"), mmap_mode="r")
        H = torch.zeros(K, K, device="cuda", dtype=torch.float32); seen = 0
        for a in range(0, n, 8192):
            X = torch.from_numpy(np.array(mm[a:min(n, a + 8192)])).view(torch.bfloat16).cuda()
            H, seen = accumulate_hessian(H, X, seen)
        res[k] = {"H": H.cpu(), "n": seen}
        del H; torch.cuda.empty_cache()
        print(f"reduced {k}: {seen} rows", flush=True)
    torch.save(res, os.path.join(OUT, "hessians.pt"))
    for f in glob.glob(os.path.join(OUT, "rows_*.npy")):
        os.remove(f)
    print("saved", os.path.join(OUT, "hessians.pt"), {k: v["n"] for k, v in res.items()})


# ---- --dump-target (drafter-2) ------------------------------------------------------------
# domain -> (source, thinking). "gen" sources are drafter/data/prompts.jsonl `src` values
# (collect_prompts.py); "real" is bench/prompts_real.jsonl, the drafter README's 8-prompt
# protocol (vllm bench serve applies the chat template with its default, thinking on);
# "labd" is bench/labd_bench.py's six tasks over its frozen corpus.
DUMP_DOMAINS = {
    "real_think": ("real", None, True), "real_nothink": ("real", None, False),
    "code": ("gen", ("magicoder",), False), "danish": ("gen", ("da-instruction", "skolegpt"), False),
    "chat": ("gen", ("ultrachat",), False), "think": ("gen", ("ultrachat", "da-reasoning"), True),
    "gsm8k_think": ("gsm8k", None, True), "gsm8k_nothink": ("gsm8k", None, False),
    "labd": ("labd", None, False),
}
LABD_TASKS = [   # bench/labd_bench.py ALL_TASKS
    ("copy", "Gengiv ordret de første 60 linjer af dokumentet. Ingen kommentarer, kun teksten."),
    ("code", "Gengiv alle bash-kommandoer fra teksten i én samlet liste, ordret."),
    ("edit", "Omskriv afsnittet om Docker så det er kortere, men behold alle kommandoer ordret."),
    ("quote", "Citér ordret de afsnit der handler om hukommelse og KV-cache, og forklar dem."),
    ("summary", "Skriv en grundig, struktureret opsummering på dansk."),
    ("qa", "Svar udførligt: hvilke optimeringer gav mest, og hvorfor?"),
]


def _dump_prompts(domains, n, skip, prompt_seed, max_tokens):
    """[(domain, src, messages, think, max_tokens)] in a fixed order."""
    import random
    out = []
    gen = None
    for dom in domains:
        if dom not in DUMP_DOMAINS:
            raise SystemExit(f"unknown domain {dom}; known: {','.join(DUMP_DOMAINS)}")
        kind, srcs, think = DUMP_DOMAINS[dom]
        if kind == "real":
            recs = [json.loads(l) for l in open(os.path.join(REPO, "bench", "prompts_real.jsonl"), encoding="utf-8")]
            for i, r in enumerate(recs[skip:]):
                out.append((dom, f"real/{skip + i}", [{"role": "user", "content": r["prompt"]}], think, 1024))
            continue
        if kind == "labd":
            corpus = os.path.expanduser(arg("--labd-corpus", "~/bench/labd_corpus.txt"))
            if not os.path.exists(corpus):
                raise SystemExit(f"labd: {corpus} missing (bench/make_long_corpus.py, or mount ~/bench)")
            doc = open(corpus, encoding="utf-8").read()[: int(arg("--labd-ctx", 16000) * 3.6)]
            for name, q in LABD_TASKS:
                out.append((dom, f"labd/{name}", [{"role": "user", "content": "Dokument:\n\n" + doc + "\n\n" + q}],
                            think, 512))
            continue
        rng = random.Random(prompt_seed)
        if kind == "gen":
            if gen is None:
                gen = [json.loads(l) for l in open(os.path.join(HERE, "data", "prompts.jsonl"), encoding="utf-8")]
            pool = [r for r in gen if r["src"] in srcs and bool(r["think"]) == think]
            rng.shuffle(pool)
            for r in pool[skip:skip + n]:
                out.append((dom, f"{r['src']}/{r['id']}", r["messages"], think, max_tokens))
        else:   # gsm8k
            import pandas as pd
            gdir = arg("--gsm8k", os.path.join(REPO, "bench", "quality-data", "gsm8k"))
            files = sorted(glob.glob(os.path.join(gdir, "**", "test-*.parquet"), recursive=True))
            if not files:
                raise SystemExit(f"gsm8k: no test-*.parquet under {gdir}")
            qs = list(pd.read_parquet(files[0])["question"])
            idx = list(range(len(qs)))
            rng.shuffle(idx)
            for i in idx[skip:skip + n]:
                out.append((dom, f"gsm8k/{i}", [{"role": "user", "content": qs[i]}], think, max_tokens))
    return out


def dump_target():
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from dflash2_ref import CaptureWriter, DFlash2Config, topk_rows
    out_dir = arg("--dump-target", "")
    target = os.environ.get("MODEL", os.path.join(REPO, "models", "Qwen3.8-27B-W4A16-AutoRound-fast"))
    dcfg = DFlash2Config.from_dir(arg("--draft-config", DRAFT))
    taps = dcfg.target_layer_ids
    window = max(w for w in dcfg.windows if w is not None) if any(dcfg.windows) else 1 << 30
    domains = arg("--domains", ",".join(DUMP_DOMAINS)).split(",")
    modes = arg("--modes", "greedy,sample").split(",")
    seeds = [int(s) for s in str(arg("--seeds", "0")).split(",")]
    n, skip, pseed = arg("--n", 12), arg("--skip", 0), arg("--prompt-seed", 0)
    TOPK = 64
    if any(m not in ("greedy", "sample") for m in modes):
        raise SystemExit(f"--modes takes greedy,sample; got {modes}")

    tok = AutoTokenizer.from_pretrained(target)
    prompts = []
    for dom, src, msgs, think, mt in _dump_prompts(domains, n, skip, pseed, arg("--max-tokens", 768)):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        prompts.append((dom, src, think, mt, tok(text, add_special_tokens=False)["input_ids"]))
    if not prompts:
        raise SystemExit("no prompts selected")
    need = max(len(p[4]) + p[3] for p in prompts) + 8
    max_len = int(os.environ.get("MAX_LEN", -(-need // 1024) * 1024))
    print(f"{len(prompts)} prompts over {domains}, modes {modes}, seeds {seeds}, max_model_len {max_len}, "
          f"taps {taps}, window {window}", flush=True)

    llm = LLM(
        model=target, served_model_name="qwen3.8-27b", enforce_eager=True,   # eager: module hooks fire
        gpu_memory_utilization=float(os.environ.get("GPU_UTIL", 0.85)),
        max_model_len=max_len, max_num_seqs=int(os.environ.get("MAX_SEQS", 16)),
        max_num_batched_tokens=int(os.environ.get("MAX_BATCHED", 8192)), attention_backend="FLASH_ATTN",
        kv_cache_dtype="bfloat16", mamba_ssm_cache_dtype="float16", language_model_only=True,
        enable_prefix_caching=False,
    )
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    model = core.model_executor.driver_worker.worker.model_runner.model
    lm = getattr(model, "language_model", model)
    inner = lm.model
    assert len(inner.layers) > max(taps), f"target has {len(inner.layers)} layers, taps {taps}"
    base = llm.get_default_sampling_params()
    sampling = {k: getattr(base, k) for k in ("temperature", "top_k", "top_p", "min_p", "repetition_penalty",
                                               "presence_penalty", "frequency_penalty")}
    print("default sampling params:", sampling, flush=True)

    # ---- 1. reference responses ---------------------------------------------------------
    reqs, params, meta = [], [], []
    for i, (dom, src, think, mt, ids) in enumerate(prompts):
        for mode in modes:
            for s in (seeds if mode == "sample" else [None]):
                if mode == "greedy":
                    p = SamplingParams(temperature=0.0, max_tokens=mt)
                else:
                    p = base.clone()
                    p.max_tokens, p.n, p.seed = mt, 1, 1000 * s + i
                reqs.append({"prompt_token_ids": ids})
                params.append(p)
                meta.append(dict(domain=dom, src=src, think=think, mode=mode, seed=s, prompt_tokens=len(ids)))
    t0 = time.time()
    outs = llm.generate(reqs, params, use_tqdm=False)
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"generated {len(outs)} responses, {ntok} tokens, {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- 2. prefill pass with hooks -----------------------------------------------------
    st = {"on": False, "keep_from": 0, "m": 0, "sel": None, "taps": {i: [] for i in taps}, "top": []}

    def pre(mod, args, kwargs):
        if not st["on"]:
            return
        pos = kwargs.get("positions", args[0] if args else None)
        pos = pos[0] if pos.dim() == 2 else pos          # M-RoPE positions are [3, N]
        ar = torch.arange(pos.shape[0], device=pos.device)
        bad = (pos != pos[0] + ar).nonzero()
        m = int(bad[0]) if bad.numel() else int(pos.shape[0])   # leading real rows; padding after
        st["m"], st["sel"] = m, pos[:m] >= st["keep_from"]

    def tap_hook(i):
        def h(mod, args, out):
            if st["on"]:
                hs, res = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
                v = hs[: st["m"]] + res[: st["m"]] if res is not None else hs[: st["m"]]
                st["taps"][i].append(v[st["sel"]].to(torch.bfloat16).cpu())
        return h

    def norm_hook(mod, args, out):
        if st["on"]:
            h = (out[0] if isinstance(out, tuple) else out)[: st["m"]][st["sel"]]
            for a in range(0, h.shape[0], 256):
                ti, tv, lse = topk_rows(lm.compute_logits(h[a:a + 256]), TOPK)
                st["top"].append((ti.cpu(), tv.cpu(), lse.cpu()))

    # The decoder model's __call__ is vLLM's support_torch_compile wrapper, which calls
    # forward() directly (no module hooks), so positions come from layer 0's call instead.
    hooks = [inner.layers[0].register_forward_pre_hook(pre, with_kwargs=True),
             inner.norm.register_forward_hook(norm_hook)]
    hooks += [inner.layers[i].register_forward_hook(tap_hook(i)) for i in taps]
    wr = CaptureWriter(out_dir, {
        "taps": taps, "hidden": int(dcfg.target_H), "dtype": "bfloat16", "topk": TOPK, "window": window,
        "target": target, "draft_config": arg("--draft-config", DRAFT), "sampling": sampling,
        "domains": domains, "modes": modes, "seeds": seeds, "n": n, "skip": skip, "prompt_seed": pseed,
        "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    one = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.time()
    agree = []
    for k, (o, mt) in enumerate(zip(outs, meta)):
        prompt_ids = list(o.prompt_token_ids)
        resp = list(o.outputs[0].token_ids)
        if len(resp) < 2:
            print(f"  skip {mt['src']} {mt['mode']}: {len(resp)} response tokens", flush=True)
            continue
        full = prompt_ids + resp
        r0 = len(prompt_ids)
        keep = max(0, r0 - window + 1)
        st.update(on=True, keep_from=keep, taps={i: [] for i in taps}, top=[])
        llm.generate([{"prompt_token_ids": full}], one, use_tqdm=False)
        st["on"] = False
        rows = len(full) - keep
        tp = torch.cat([torch.cat(st["taps"][i]) for i in taps], -1)
        ti = torch.cat([x[0] for x in st["top"]])
        tv = torch.cat([x[1] for x in st["top"]])
        lse = torch.cat([x[2] for x in st["top"]])
        if tp.shape[0] != rows or ti.shape[0] != rows:
            raise RuntimeError(f"{mt['src']}: captured {tp.shape[0]}/{ti.shape[0]} rows, expected {rows} "
                               f"(padding or chunking the hooks did not account for)")
        ids = torch.tensor(full[keep:], dtype=torch.int32)
        nxt = ids[r0 - keep:].long()
        top1 = ti[r0 - keep - 1:-1, 0]
        a = float((top1 == nxt).float().mean())
        if mt["mode"] == "greedy":
            agree.append(a)
        wr.add(dict(mt, prompt_len=r0 - keep, start_pos=keep, resp_len=len(resp),
                    finish=o.outputs[0].finish_reason, top1_agree=round(a, 4)), tp, ids, ti, tv, lse)
        if k % 16 == 0:
            print(f"[{k + 1}/{len(outs)}] {mt['domain']} {mt['mode']} rows {rows} top1-agree {a:.3f} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
    for h in hooks:
        h.remove()
    wr.close()
    print(f"wrote {out_dir}: {wr.n_seq} sequences, {wr.n_tok} rows "
          f"({wr.n_tok * len(taps) * dcfg.target_H * 2 / 2**30:.1f} GiB taps); greedy top-1 vs reference "
          f"agreement {np.mean(agree) if agree else float('nan'):.4f} (decode vs prefill numerics; "
          f"eval_dflash2.py calls the capture INVALID below 0.97)", flush=True)



if __name__ == "__main__":
    if "--dump-target" in sys.argv:
        dump_target()
    elif "--reduce-only" in sys.argv:
        reduce_wide()
    else:
        main()
