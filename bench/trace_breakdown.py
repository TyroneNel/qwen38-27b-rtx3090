"""Per-step GPU time attribution from a vLLM torch-profiler trace (DeepInfer attn-2 / evals-8).

What it reads. The trace vLLM 0.31 writes when the server runs with
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"DIR"}'
and gets POST /start_profile ... POST /stop_profile. Each GPU worker writes
DIR/<worker_name>.<time_ns>.pt.trace.json.gz (torch.profiler.tensorboard_trace_handler; worker_name
is dp0_pp0_tp0_dcp0_ep0_rank0 style, prefixed with the profile prefix if one was given) with CPU and
CUDA activities and, by default, Python stacks (torch_profiler_with_stack=true). The API process
writes <host>_<pid>.async_llm.<ns>.pt.trace.json.gz (CPU only) and the worker profiler_out_0.txt
(key_averages table); both are skipped here. Given DIR, the newest worker trace is read (--all reads
every one); given a file, that file.

Steps. While the profiler runs, the worker wraps each execute_model in a record_function range
named execute_context_<n>(<tok>)_generation_<m>(<tok>) (gpu_worker.annotate_profile; with
detailed_trace_annotation the generation part also carries sk<sum of seq lens>). One engine step
is the CPU time from one execute_ range start to the next: it also holds sample_tokens (target
logits, rejection sampling, the drafter's propose), which run after the execute_ range closes.
Every GPU kernel is put in the step whose range was open when the CPU launched it (kernel ->
cudaLaunchKernel / cuLaunchKernel / cudaGraphLaunch by correlation id; kernels replayed from a CUDA
graph carry the graph launch's id). By default only pure decode steps are kept (no context-phase
request). Without execute_ ranges (CPU activity off), --step-marker REGEX names the event that
starts a step, else steps are cut at GPU idle gaps longer than --gap-us.

Categories. A kernel launched under the drafter (a Python frame or range matching
spec_decode/dflash*/speculator*/propose, or vLLM's "gpu_model_runner: draft" scope) is "drafter",
with its own breakdown by kernel name. Else one launched under compute_logits / LogitsProcessor is
"lm_head_logits", else one under the sampler / rejection sampler is "sampling". Everything else is
classified by kernel name, first match wins:
  full_attn        pthfa (pth_attn::attn_dec/attn_fwd/attn_fwd_window, attn_combine), FA2 and the
                   fp8-KV FA2 plugin (flash_fwd*, fa2_fp8kv), KVarN Triton (_kvarn_*, minus the
                   scatter store), spec-decode split-KV (_spec_attn_partial/_combine), Triton
                   unified / decode attention, merge_attn_states, flashinfer, int4/turboquant
  kv_cache_write   reshape_and_cache*, _kvarn_scatter_store_kernel, kv-cache store kernels
  gdn              fla fused_recurrent*, fused_sigmoid_gating*, chunk_*, solve_tril, l2norm,
                   layer_norm_gated, causal_conv1d*, mamba state copies / align
  linear_gemm      marlin, humming (+ its process_input quant), cutlass, cuBLAS (gemm/gemv/xmma/
                   s16816), triton w4a16 / scaled_mm, int8/fp8 activation quant
  sampling         rejection*, sample*, top-k/top-p, gumbel, argmax, softmax, logprobs, penalties
  norm_elementwise rms/layer norm, act_and_mul / silu, rotary, torch elementwise / reduce / copy /
                   index / cat, memcpy and memset
  bookkeeping      runner input prep: block tables, slot mappings, positions, post-update
  other            nothing matched: listed one by one with time, so a new kernel never hides.
Without Python stacks, the lm_head is guessed per step as the longest linear_gemm launch when it
is over 2x the next one (--no-lmhead-guess turns that off); the output says which source was used.

Per step: wall is the GPU spacing from a step's first kernel to the next step's first kernel;
busy is the union of all kernel/memcpy/memset intervals in that window over the wall; idle is
wall - busy (host gaps and launch latency); gap is the idle stretch between the step's last kernel
and the next step's first. cpu_gap is the time between an execute_ range end and the next one's
start (sample_tokens, drafter launches, scheduler). The whole-window busy fraction is the union over
first-kernel-start .. last-kernel-end. Kernel time sums every kernel, so it can exceed busy when
streams overlap.

Usage:
  python3 bench/trace_breakdown.py DIR_OR_TRACE [--all] [--all-steps] [--skip N] [--top K]
      [--json OUT.json|-] [--step-marker REGEX] [--gap-us US] [--per-step]

Exit: 0 measurement printed; 2 no trace found, no GPU kernel in it, or no step to report.
CPU, Python 3 standard library only.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------- classification tables

# Launch scopes from the CPU stack (Python frames, record_function ranges, aten ops) that encloses
# the launch. Priority: drafter > lm_head_logits > sampling (the V2 runner's sample() contains
# compute_logits; the drafter's propose calls the target's compute_logits and its own sampler).
SCOPE_RULES: list[tuple[str, re.Pattern]] = [
    ("drafter", re.compile(
        r"spec_decode/(?!rejection_sampler)|speculator|dflash|drafter|gpu_model_runner: draft"
        r"|\bpropose\b|proposer|_generate_draft|_chain_generate", re.I)),
    ("lm_head_logits", re.compile(
        r"compute_logits|logits_processor|LogitsProcessor|\blm_head\b|ParallelLMHead", re.I)),
    ("sampling", re.compile(
        r"rejection_sampler|RejectionSampler|/sample/|sampler\.py|gpu_model_runner: sample\b"
        r"|\): sample$|\): sample_recovered|\bSampler\b", re.I)),
]

# Kernel-name rules, first match wins. Attention goes before GEMMs: FA2 template arguments
# contain cutlass:: and cute:: types.
NAME_RULES: list[tuple[str, re.Pattern]] = [
    ("drafter", re.compile(
        r"dflash|_selector_walk_kernel|_cache_draft_logits|_suffix_lookup_kernel|_fuse_draft_kernel"
        r"|_point_mass_draft_logits|eagle_|_prepare_dflash_inputs|copy_and_expand_(eagle|dflash)",
        re.I)),
    ("kv_cache_write", re.compile(
        r"reshape_and_cache|_kvarn_scatter_store|_reshape_cache|kv_cache_scatter|_tq_fused_store"
        r"|_store_quantized_value|concat_and_cache|_k_quant_kernel", re.I)),
    ("full_attn", re.compile(
        r"pth_attn|pthfa|attn_dec_kernel|attn_fwd_kernel|attn_fwd_window|attn_combine"
        r"|flash_fwd|fa2_fp8kv|flash::|compute_attn|combine_attn_seqk"
        r"|kvarn|_spec_attn|kernel_unified_attention|unified_attention|reduce_segments"
        r"|triton_attn|paged_attention|_fwd_kernel_stage|_fwd_grouped_kernel_stage"
        r"|merge_attn_states|flashinfer|BatchDecode|BatchPrefill|SinglePrefill|SingleDecode"
        r"|_prefill_attn_kernel|_k_stats_kernel|_tq_decode|_tq_full_dequant|kernel_tq_unified"
        r"|_attn_packed|_hadamard_mma|^_fwd_kernel$|_fwd_kernel_alibi", re.I)),
    ("gdn", re.compile(
        r"fused_recurrent|gated_delta|sigmoid_gating|gdn|chunk_(gated|local|scaled|fwd|kda|gla)"
        r"|solve_tril|merge_16x16|recompute_w_u|l2norm|layer_norm_gated|_layer_norm_fwd_1pass"
        r"|causal_conv1d|selective_(scan|state)|mamba|_ssd|ssd_|state_passing|_scatter_states"
        r"|_gather_initial_states|_post_conv|recoverssm|replayssm|\bkda|_kda|batch_memcpy_kernel"
        r"|aligned_state_indices|_memcpy_u64_tiled", re.I)),
    ("linear_gemm", re.compile(
        r"marlin|humming|process_input_kernel|finalize_group_token_scales|cutlass|gemm|gemv"
        r"|xmma|s16816|s1688|ampere_|sm80_|sm86_|sm89_|cublas|splitKreduce|matmul|_mm_kernel"
        r"|scaled_mm|w4a16|w8a8|exllama|gptq|awq|wvSplitK|LLGemm"
        r"|quant_int8|int8_quant|scaled_int8|scaled_fp8|fp8_quant|per_token_quant"
        r"|per_token_group_quant|act_quant", re.I)),
    ("sampling", re.compile(
        r"rejection|sampl|topk|top_k|topp|top_p|gumbel|argmax|softmax|logsumexp|_compute_global"
        r"|_compute_local|resample|_penalties|_temperature|_min_p|bad_words|logprob|_ranks_kernel"
        r"|_flatten_sampled|_gather_draft_sampled|cumulative_log_p|residual_mass|_bias_kernel"
        r"|_compute_max_and_sumexp|_insert_resampled|expand_kernel|radixSort|DeviceRadixSort"
        r"|DeviceSegmentedRadixSort|bitonicSort|gatherTopK|sort_", re.I)),
    ("bookkeeping", re.compile(
        r"_prepare_|_post_update|_combine_sampled|_expand_idx|block_table|slot_mapping"
        r"|_token_request|_gather_block_tables|_compute_slot|_apply_write_kernel|_zero_kv_blocks"
        r"|_get_num_sampled|_scatter_num_accepted|_fill_num_accepted|_dcp_local_seq_lens"
        r"|_num_nans_kernel|_compact_sampling_mask|_apply_grammar_bitmask|_structured", re.I)),
    ("norm_elementwise", re.compile(
        r"norm|rms|silu|gelu|act_and_mul|swiglu|activation|rotary|rope|elementwise|vectorized"
        r"|unrolled|reduce_kernel|reduce|copy|memcpy|memset|fill|CatArray|cat_|index|gather"
        r"|scatter|embedding|where|cast|_to_copy|clamp|Memcpy|Memset|arange|cumsum|compare"
        r"|masked|div|mul|add|sub|exp|neg|bitwise", re.I)),
]

CATEGORY_ORDER = [
    "full_attn", "kv_cache_write", "gdn", "linear_gemm", "lm_head_logits", "drafter",
    "sampling", "norm_elementwise", "bookkeeping", "other",
]

CPU_CATS = {"cpu_op", "user_annotation", "python_function"}
LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
EXEC_RE = re.compile(r"^execute_")
CTX_RE = re.compile(r"_context_(\d+)\(")
GEN_RE = re.compile(r"_generation_(\d+)\(([^)]*)\)")
SK_RE = re.compile(r"sk(\d+)")


def classify_name(name: str, cat: str) -> str:
    if cat in ("gpu_memcpy", "gpu_memset"):
        return "norm_elementwise"
    for label, rx in NAME_RULES:
        if rx.search(name):
            return label
    return "other"


def short_name(name: str, width: int = 96) -> str:
    """Kernel name without the template/argument tail, for tables."""
    s = name
    if s.startswith("void "):
        s = s[5:]
    cut = len(s)
    for ch in "<(":
        i = s.find(ch)
        if 0 <= i < cut:
            cut = i
    base = s[:cut] if cut > 0 else s
    if base != s:
        base += "<..>" if s[cut] == "<" else "(..)"
    return base if len(base) <= width else base[: width - 3] + "..."


# ---------------------------------------------------------------- trace loading

def find_traces(path: Path, take_all: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    cands = [p for p in path.iterdir()
             if p.is_file() and re.search(r"\.pt\.trace\.json(\.gz)?$", p.name)
             and ".async_llm." not in p.name]
    cands.sort(key=lambda p: p.stat().st_mtime)
    if not cands:
        return []
    return cands if take_all else [cands[-1]]


def load_events(path: Path) -> list[dict]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("traceEvents", [])


def corr_of(ev: dict):
    a = ev.get("args") or {}
    c = a.get("correlation")
    if c is None:
        c = a.get("correlationId")
    return c


# ---------------------------------------------------------------- analysis

def union_len(intervals: list[tuple[float, float]], lo: float | None = None,
              hi: float | None = None) -> float:
    """Total length of the union of [s, e) intervals, optionally clipped to [lo, hi)."""
    if lo is not None:
        intervals = [(max(s, lo), min(e, hi)) for s, e in intervals if e > lo and s < hi]
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total, cs, ce = 0.0, intervals[0][0], intervals[0][1]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    return total + (ce - cs)


def resolve_launch_scopes(events: list[dict]):
    """Per launch correlation id: (cpu ts, launch name, scope, had_python_stack).

    Sweeps each CPU thread's interval events with a stack; a launch's scope is the
    highest-priority SCOPE_RULES label matched by any enclosing event name.
    """
    by_tid: dict[tuple, list[tuple]] = defaultdict(list)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        if cat in CPU_CATS:
            by_tid[(ev.get("pid"), ev.get("tid"))].append(
                (float(ev["ts"]), -float(ev.get("dur", 0)), 0, ev))
        elif cat in LAUNCH_CATS and corr_of(ev) is not None:
            by_tid[(ev.get("pid"), ev.get("tid"))].append(
                (float(ev["ts"]), -float(ev.get("dur", 0)), 1, ev))
    scope_cache: dict[str, int] = {}

    def scope_idx(name: str) -> int:
        r = scope_cache.get(name)
        if r is None:
            r = len(SCOPE_RULES)
            for i, (_, rx) in enumerate(SCOPE_RULES):
                if rx.search(name):
                    r = i
                    break
            scope_cache[name] = r
        return r

    launches: dict = {}
    for evs in by_tid.values():
        evs.sort(key=lambda t: (t[0], t[1], t[2]))
        stack: list[tuple[float, int, bool]] = []  # (end, best scope idx so far, any python frame)
        for ts, negdur, kind, ev in evs:
            while stack and stack[-1][0] <= ts:
                stack.pop()
            parent_scope = stack[-1][1] if stack else len(SCOPE_RULES)
            parent_py = stack[-1][2] if stack else False
            if kind == 0:
                own = scope_idx(ev.get("name", ""))
                stack.append((ts - negdur, min(parent_scope, own),
                              parent_py or ev.get("cat") == "python_function"))
            else:
                sc = SCOPE_RULES[parent_scope][0] if parent_scope < len(SCOPE_RULES) else None
                launches[corr_of(ev)] = (ts, ev.get("name", ""), sc, parent_py)
    return launches


def derive_steps(events, kernels, launches, marker: re.Pattern | None, gap_us: float):
    """Returns (source, steps). A step: dict(start, end, name, by) with by in {'cpu', 'gpu'}:
    kernels are assigned by CPU launch time for 'cpu' steps and by GPU start for 'gpu' steps."""
    if marker is None:
        ex = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", ""))
                    for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation"
                    and EXEC_RE.match(e.get("name", "")))
        if ex:
            return "execute_ ranges", [dict(start=s, end=e, name=n, by="cpu") for s, e, n in ex]
    else:
        ms = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", ""),
                     e.get("cat", "")) for e in events
                    if e.get("ph") == "X" and marker.search(e.get("name", "")))
        if ms:
            by = "gpu" if all(m[3] in GPU_CATS or m[3] == "gpu_user_annotation" for m in ms) else "cpu"
            if by == "cpu":
                ms = [m for m in ms if m[3] not in GPU_CATS and m[3] != "gpu_user_annotation"]
            return (f"--step-marker {marker.pattern!r}",
                    [dict(start=s, end=e, name=n, by=by) for s, e, n, _ in ms])
        return f"--step-marker {marker.pattern!r} (no match)", []
    # GPU idle-gap segmentation.
    ks = sorted(kernels, key=lambda k: k["ts"])
    steps, cur_start, max_end = [], None, None
    for k in ks:
        if cur_start is None:
            cur_start, max_end = k["ts"], k["end"]
            continue
        if k["ts"] - max_end > gap_us:
            steps.append(dict(start=cur_start, end=max_end, name="gap-step", by="gpu"))
            cur_start = k["ts"]
        max_end = max(max_end, k["end"])
    if cur_start is not None:
        steps.append(dict(start=cur_start, end=max_end, name="gap-step", by="gpu"))
    return f"GPU idle gaps > {gap_us:g} us", steps


def step_kind(name: str):
    """(is_decode, n_gen_tokens, sum_seq_len) from an execute_ annotation, None where unknown."""
    m = CTX_RE.search(name)
    g = GEN_RE.search(name)
    if not m or not g:
        return None, None, None
    is_decode = int(m.group(1)) == 0 and int(g.group(1)) > 0
    inner = g.group(2)
    sk = SK_RE.search(inner)
    tok = re.match(r"(?:sq)?(\d+)", inner)
    return is_decode, (int(tok.group(1)) if tok else None), (int(sk.group(1)) if sk else None)


def med(xs):
    return statistics.median(xs) if xs else 0.0


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def analyze(path: Path, args) -> dict | None:
    events = load_events(path)
    kernels = []
    for ev in events:
        if ev.get("ph") == "X" and ev.get("cat") in GPU_CATS:
            ts = float(ev["ts"])
            kernels.append(dict(name=ev.get("name", ""), cat=ev["cat"], ts=ts,
                                end=ts + float(ev.get("dur", 0)), corr=corr_of(ev),
                                stream=(ev.get("args") or {}).get("stream")))
    if not kernels:
        return None
    launches = resolve_launch_scopes(events)
    marker = re.compile(args.step_marker) if args.step_marker else None
    source, steps = derive_steps(events, kernels, launches, marker, args.gap_us)

    # Classify.
    linked = py_linked = 0.0
    for k in kernels:
        ln = launches.get(k["corr"])
        k["launch_ts"] = ln[0] if ln else None
        k["graph"] = bool(ln and "Graph" in ln[1])
        scope = ln[2] if ln else None
        dur = k["end"] - k["ts"]
        if ln:
            linked += dur
            if ln[3]:
                py_linked += dur
        nc = classify_name(k["name"], k["cat"])
        k["namecat"] = nc
        if scope == "drafter" or nc == "drafter":
            k["category"], k["sub"] = "drafter", nc
        elif scope in ("lm_head_logits", "sampling"):
            k["category"], k["sub"] = scope, nc
        else:
            k["category"], k["sub"] = nc, nc
        k["lmhead_by"] = "stack" if scope == "lm_head_logits" else None
    total_k = sum(k["end"] - k["ts"] for k in kernels)

    # Assign kernels to steps.
    starts = [s["start"] for s in steps]
    per_step: list[list[dict]] = [[] for _ in steps]
    unassigned = 0
    for k in kernels:
        if not steps:
            break
        by = steps[0]["by"]
        t = k["launch_ts"] if by == "cpu" and k["launch_ts"] is not None else k["ts"]
        i = bisect.bisect_right(starts, t) - 1
        if i < 0:
            unassigned += 1
            continue
        per_step[i].append(k)

    # lm_head guess when no stack ever named compute_logits.
    lm_source = "stack (compute_logits / LogitsProcessor frames)"
    if not any(k["category"] == "lm_head_logits" for k in kernels):
        if args.no_lmhead_guess:
            lm_source = "none (no compute_logits frame; guess disabled)"
        else:
            lm_source = "guess: per step the longest linear_gemm launch, if over 2x the next one"
            for ks in per_step:
                g = sorted((k for k in ks if k["category"] == "linear_gemm"),
                           key=lambda k: k["end"] - k["ts"], reverse=True)
                if len(g) >= 2 and (g[0]["end"] - g[0]["ts"]) > 2 * (g[1]["end"] - g[1]["ts"]):
                    g[0]["category"], g[0]["lmhead_by"] = "lm_head_logits", "guess"

    # Step selection.
    sel = []
    n_decode = 0
    for i, s in enumerate(steps):
        is_dec, ntok, sk = step_kind(s["name"]) if source == "execute_ ranges" else (None, None, None)
        s["is_decode"], s["gen_tokens"], s["seq_len_sum"] = is_dec, ntok, sk
        if is_dec:
            n_decode += 1
        if not per_step[i]:
            continue
        if not args.all_steps and is_dec is False:
            continue
        sel.append(i)
    sel = sel[args.skip:]
    if not sel:
        return dict(trace=str(path), error="no step to report", steps_source=source,
                    n_steps_total=len(steps))

    all_iv = [(k["ts"], k["end"]) for k in kernels]
    first_k = [min(k["ts"] for k in ks) if ks else None for ks in per_step]

    def next_first(i):
        for j in range(i + 1, len(steps)):
            if first_k[j] is not None:
                return first_k[j]
        return None

    cat_us = {c: [] for c in CATEGORY_ORDER}
    draft_sub_us: dict[str, list[float]] = defaultdict(list)
    kern_tot: dict[tuple, list] = {}  # (category, short) -> [us, calls, full name]
    step_rows = []
    walls, busies, idles, gaps, cpu_gaps, ksums = [], [], [], [], [], []
    for i in sel:
        ks = per_step[i]
        per_cat = defaultdict(float)
        per_sub = defaultdict(float)
        for k in ks:
            d = k["end"] - k["ts"]
            per_cat[k["category"]] += d
            if k["category"] == "drafter":
                per_sub[k["sub"]] += d
            key = (k["category"], short_name(k["name"]))
            ent = kern_tot.setdefault(key, [0.0, 0, k["name"]])
            ent[0] += d
            ent[1] += 1
        for c in CATEGORY_ORDER:
            cat_us[c].append(per_cat.get(c, 0.0))
        for c in CATEGORY_ORDER:
            draft_sub_us[c].append(per_sub.get(c, 0.0))
        ksum = sum(per_cat.values())
        ksums.append(ksum)
        row = dict(step=i, name=steps[i]["name"], kernel_ms=ksum / 1e3, kernels=len(ks),
                   graph_kernels=sum(1 for k in ks if k["graph"]),
                   seq_len_sum=steps[i].get("seq_len_sum"), gen_tokens=steps[i].get("gen_tokens"),
                   by_category_ms={c: per_cat[c] / 1e3 for c in CATEGORY_ORDER if per_cat.get(c)})
        nf = next_first(i)
        if nf is not None:
            wall = nf - first_k[i]
            busy = union_len(all_iv, first_k[i], nf)
            last_end = max(k["end"] for k in ks)
            walls.append(wall)
            busies.append(busy)
            idles.append(wall - busy)
            gaps.append(max(0.0, nf - last_end))
            row.update(wall_ms=wall / 1e3, busy_frac=busy / wall if wall > 0 else None,
                       gap_ms=max(0.0, nf - last_end) / 1e3)
        if steps[i]["by"] == "cpu" and i + 1 < len(steps):
            cg = steps[i + 1]["start"] - steps[i]["end"]
            cpu_gaps.append(cg)
            row["cpu_gap_ms"] = cg / 1e3
        step_rows.append(row)

    n = len(sel)
    sel_k = [k for i in sel for k in per_step[i]]
    k_sel_total = sum(k["end"] - k["ts"] for k in sel_k)
    w_lo = min(k["ts"] for k in kernels)
    w_hi = max(k["end"] for k in kernels)
    mean_wall = mean(walls)

    cats = {}
    for c in CATEGORY_ORDER:
        xs = cat_us[c]
        tot = sum(xs)
        top = sorted(((v[0], v[1], name, v[2]) for (cc, name), v in kern_tot.items() if cc == c),
                     reverse=True)
        cats[c] = dict(
            ms_per_step_mean=mean(xs) / 1e3, ms_per_step_median=med(xs) / 1e3,
            pct_of_kernel_time=100 * tot / k_sel_total if k_sel_total else 0.0,
            pct_of_step_wall=100 * mean(xs) / mean_wall if mean_wall else None,
            calls_per_step=sum(t[1] for t in top) / n,
            top=[dict(kernel=nm, ms_per_step=us / n / 1e3, calls_per_step=calls / n,
                      full_name=full if full != nm else None)
                 for us, calls, nm, full in top[: args.top]])
    # Every kernel no name rule matched, wherever its launch scope put it.
    unk: dict[tuple, list] = {}
    for k in sel_k:
        if k["namecat"] == "other":
            ent = unk.setdefault((short_name(k["name"]), k["category"]), [0.0, 0, k["name"]])
            ent[0] += k["end"] - k["ts"]
            ent[1] += 1
    unknown = [dict(kernel=nm, counted_in=cat, ms_total=v[0] / 1e3, ms_per_step=v[0] / n / 1e3,
                    calls=v[1], full_name=v[2])
               for (nm, cat), v in sorted(unk.items(), key=lambda kv: -kv[1][0])]

    return dict(
        trace=str(path),
        steps_source=source,
        n_steps_total=len(steps),
        n_steps_decode=n_decode if source == "execute_ ranges" else None,
        n_steps_used=n,
        step_filter=("all steps" if args.all_steps
                     else "decode-only (no context-phase request)" if source == "execute_ ranges"
                     else "all steps (kind unknown without execute_ ranges)"),
        skipped_first=args.skip,
        kernels_before_first_step=unassigned,
        gen_tokens_per_step_median=med([r["gen_tokens"] for r in step_rows
                                        if r["gen_tokens"] is not None]) or None,
        seq_len_sum_median=med([r["seq_len_sum"] for r in step_rows
                                if r["seq_len_sum"] is not None]) or None,
        launch_link_frac=linked / total_k if total_k else 0.0,
        python_stack_frac=py_linked / total_k if total_k else 0.0,
        lm_head_source=lm_source,
        window=dict(wall_ms=(w_hi - w_lo) / 1e3, busy_ms=union_len(all_iv) / 1e3,
                    busy_frac=union_len(all_iv) / (w_hi - w_lo) if w_hi > w_lo else None,
                    kernel_ms=total_k / 1e3, n_kernels=len(kernels)),
        per_step=dict(
            wall_ms_mean=mean(walls) / 1e3, wall_ms_median=med(walls) / 1e3,
            kernel_ms_mean=mean(ksums) / 1e3, kernel_ms_median=med(ksums) / 1e3,
            busy_ms_mean=mean(busies) / 1e3,
            busy_frac=sum(busies) / sum(walls) if walls and sum(walls) > 0 else None,
            idle_ms_mean=mean(idles) / 1e3, idle_ms_median=med(idles) / 1e3,
            gap_ms_mean=mean(gaps) / 1e3, gap_ms_median=med(gaps) / 1e3,
            cpu_gap_ms_mean=mean(cpu_gaps) / 1e3 if cpu_gaps else None,
            cpu_gap_ms_median=med(cpu_gaps) / 1e3 if cpu_gaps else None,
            n_with_wall=len(walls)),
        categories=cats,
        drafter_breakdown={c: dict(ms_per_step_mean=mean(v) / 1e3, ms_per_step_median=med(v) / 1e3)
                           for c, v in draft_sub_us.items() if sum(v) > 0},
        unknown_kernels=unknown,
        steps=step_rows if args.per_step else None,
    )


# ---------------------------------------------------------------- output

def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def print_report(r: dict, top: int) -> None:
    print(f"trace: {r['trace']}")
    if r.get("error"):
        print(f"  {r['error']} (steps from {r['steps_source']}, {r['n_steps_total']} found)")
        return
    dec = f", {r['n_steps_decode']} decode" if r["n_steps_decode"] is not None else ""
    print(f"steps: {r['steps_source']}: {r['n_steps_total']} found{dec}; used {r['n_steps_used']} "
          f"({r['step_filter']}, first {r['skipped_first']} skipped)")
    extra = []
    if r["gen_tokens_per_step_median"]:
        extra.append(f"generation tokens/step median {r['gen_tokens_per_step_median']:g}")
    if r["seq_len_sum_median"]:
        extra.append(f"sum seq len median {r['seq_len_sum_median']:g}")
    if extra:
        print("       " + "; ".join(extra))
    print(f"links: {100 * r['launch_link_frac']:.1f}% of kernel time tied to a CPU launch, "
          f"{100 * r['python_stack_frac']:.1f}% under a Python stack; lm_head: {r['lm_head_source']}")
    w = r["window"]
    print(f"window: {w['wall_ms']:.2f} ms wall, {w['n_kernels']} GPU events, kernel sum "
          f"{w['kernel_ms']:.2f} ms, busy {w['busy_ms']:.2f} ms = {fmt(w['busy_frac'] and 100 * w['busy_frac'], 1)}%")
    p = r["per_step"]
    print(f"per step (n={p['n_with_wall']} with a successor): wall {fmt(p['wall_ms_mean'])} mean / "
          f"{fmt(p['wall_ms_median'])} median ms; kernel sum {fmt(p['kernel_ms_mean'])} ms; GPU busy "
          f"{fmt(p['busy_frac'] and 100 * p['busy_frac'], 1)}%; idle {fmt(p['idle_ms_mean'])} ms "
          f"(gap to next step {fmt(p['gap_ms_median'])} median); host gap between execute_ ranges "
          f"{fmt(p['cpu_gap_ms_median'])} ms median")
    print()
    print(f"{'category':<18}{'ms/step':>9}{'median':>9}{'% kern':>8}{'% wall':>8}{'calls':>8}")
    for c in CATEGORY_ORDER:
        d = r["categories"][c]
        print(f"{c:<18}{d['ms_per_step_mean']:>9.3f}{d['ms_per_step_median']:>9.3f}"
              f"{d['pct_of_kernel_time']:>8.1f}{fmt(d['pct_of_step_wall'], 1):>8}"
              f"{d['calls_per_step']:>8.1f}")
    if r["drafter_breakdown"]:
        print("\ndrafter by kernel class (ms/step mean / median):")
        for c, d in sorted(r["drafter_breakdown"].items(), key=lambda kv: -kv[1]["ms_per_step_mean"]):
            print(f"  {c:<18}{d['ms_per_step_mean']:>9.3f}{d['ms_per_step_median']:>9.3f}")
    print(f"\ntop kernels per category (ms/step, calls/step), up to {top}:")
    for c in CATEGORY_ORDER:
        tops = r["categories"][c]["top"]
        if not tops or c == "other":
            continue
        print(f"  [{c}]")
        for t in tops:
            print(f"    {t['ms_per_step']:>8.3f} {t['calls_per_step']:>6.1f}  {t['kernel']}")
    print(f"\nkernels no name rule matched, all {len(r['unknown_kernels'])} "
          "(counted in 'other' unless a launch scope claimed them):")
    for u in r["unknown_kernels"]:
        print(f"    {u['ms_per_step']:>8.3f} ms/step {u['calls']:>6} calls  "
              f"[{u['counted_in']}]  {u['kernel']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Per-step GPU time attribution from a vLLM torch-profiler trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exit 0: report printed. Exit 2: no trace, no GPU kernel, or no step to report.")
    ap.add_argument("path", help="torch_profiler_dir, or one *.pt.trace.json[.gz] file")
    ap.add_argument("--all", action="store_true",
                    help="every worker trace in the directory, not only the newest")
    ap.add_argument("--all-steps", action="store_true",
                    help="keep steps with a context-phase (prefill) request too")
    ap.add_argument("--skip", type=int, default=0, help="drop the first N selected steps")
    ap.add_argument("--top", type=int, default=8, help="kernels listed per category (default 8)")
    ap.add_argument("--step-marker", metavar="REGEX",
                    help="event name that starts a step, instead of the execute_ ranges")
    ap.add_argument("--gap-us", type=float, default=100.0,
                    help="GPU idle gap that splits steps when the trace has no step ranges "
                         "(default 100)")
    ap.add_argument("--no-lmhead-guess", action="store_true",
                    help="without compute_logits frames, leave the lm_head GEMM in linear_gemm")
    ap.add_argument("--per-step", action="store_true", help="include every step in the JSON")
    ap.add_argument("--json", metavar="OUT", help="write the result as JSON to OUT ('-' = stdout)")
    args = ap.parse_args()

    traces = find_traces(Path(args.path), args.all)
    if not traces:
        print(f"no *.pt.trace.json[.gz] worker trace at {args.path}", file=sys.stderr)
        return 2
    results, ok = [], False
    for t in traces:
        r = analyze(t, args)
        if r is None:
            print(f"trace: {t}\n  no GPU kernel events (was CUDA activity profiled?)", file=sys.stderr)
            continue
        results.append(r)
        if args.json != "-":
            print_report(r, args.top)
            print()
        ok = ok or not r.get("error")
    if args.json:
        payload = results[0] if len(results) == 1 else results
        text = json.dumps(payload, indent=1)
        if args.json == "-":
            print(text)
        else:
            Path(args.json).write_text(text + "\n", encoding="utf-8")
            print(f"json: {args.json}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
