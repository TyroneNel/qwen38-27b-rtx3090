"""Weight, head and KV-geometry arithmetic for docs/research/01-project/{quantisation,heads-drafters}.md.

CPU only. Reads safetensors *headers* from the Hugging Face Hub by HTTP range request (no
weights are downloaded), pinned to the commits below, then prints:
  1. bytes per tensor category, per repo (measured: header data_offsets)
  2. MTP and DFlash2 parameter counts per component (measured: header shapes)
  3. estimated sizes of the int8 heads prepare/ writes (formula: rows*cols + rows*cols/128*2)
  4. estimated resident weights per serving variant
  5. KV page geometry from the launcher's formula (single-user/start_qwen.sh:566-616), checked
     against the pools bench/test_resident_print.sh asserts
  6. effective bytes per pool token from documented KV_MEM pins and logged pools

  python3 -I weights_breakdown.py [--cache headers.json]   > weights_breakdown.out
"""
import json, math, re, struct, sys, urllib.request, collections

REPOS = {
    "dbirks/Qwen3.8-27B-W4A16-AutoRound": ("1f05c441c4e64ae0549de44fa9ea5a6d43610314",
        [f"model-0000{i}-of-00007.safetensors" for i in range(1, 8)] + ["model_extra_tensors.safetensors"]),
    "syvai/qwen3.8-27b-3090-fast-variant": ("124c14e7e8c7d2f5402933b9af368e772a9fcf0c",
        ["model-00007-of-00007.safetensors", "model_extra_tensors.safetensors"]),
    "incoai/Qwen3.8-27B-DFlash2": ("015e795645c74b1a0eeef3b570031fb62e769bc5", ["model.safetensors"]),
    "syvai/Qwen3.8-27B-DFlash2-W4A16": ("4d30ec736ffc6b8688dc2ae2b502d9b48bdec279", ["model.safetensors"]),
}
GiB = 2 ** 30


def header(url):
    r = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    n = struct.unpack("<Q", urllib.request.urlopen(r, timeout=60).read())[0]
    r = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + n}"})
    return json.loads(urllib.request.urlopen(r, timeout=60).read())


def load(cache):
    if cache:
        return json.load(open(cache))
    return {f"{repo}/{f}": header(f"https://huggingface.co/{repo}/resolve/{sha}/{f}")
            for repo, (sha, files) in REPOS.items() for f in files}


def nb(m): return m["data_offsets"][1] - m["data_offsets"][0]


def cat(k):
    if k.startswith("mtp."):
        return "mtp." + ("draft_lm_head" if "draft_lm_head" in k else "fc" if k.startswith("mtp.fc")
                         else "linears" if "_proj." in k else "norms")
    if "visual" in k: return "vision tower (bf16)"
    if "lm_head" in k: return "lm_head"
    if "embed_tokens" in k: return "embed_tokens"
    if re.search(r"weight_(packed|scale|shape|zero_point)$", k): return "body linears int4 (packed+scales)"
    if "linear_attn" in k:
        return "gdn in_proj_a/b (bf16)" if ("in_proj_a" in k or "in_proj_b" in k) else "gdn conv1d/A_log/dt/norm"
    return "norms/other"


def dflash_cat(k):
    k = re.sub(r"\.(weight_packed|weight_scale|weight_shape|weight)$", "", k)
    if k.startswith("candidate_selector"): return "selector " + k.split(".")[1]
    if k.startswith("fc"): return "fc (5 taps x 5120 -> 5120)"
    m = re.match(r"layers\.\d+\.(self_attn|mlp|attention_conv|mlp_conv)\.", k)
    if m: return {"self_attn": "5 layers attn q/k/v/o", "mlp": "5 layers mlp",
                  "attention_conv": "5 layers dynamic conv (attn)", "mlp_conv": "5 layers dynamic conv (mlp)"}[m.group(1)]
    return "norms"


def params(m, k):
    if k.endswith("weight_packed"): return m["shape"][0] * m["shape"][1] * 8  # int32 holds 8 int4
    if k.endswith(("weight_scale", "weight_shape")): return 0
    return math.prod(m["shape"])


def main():
    cache = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else None
    H = load(cache)
    print("# 1. bytes per category (measured from safetensors headers)")
    sizes = {}
    for repo in REPOS:
        tot = collections.Counter()
        for f, hdr in H.items():
            if not f.startswith(repo + "/"): continue
            for k, m in hdr.items():
                if k != "__metadata__": tot[dflash_cat(k) if "DFlash2" in repo else cat(k)] += nb(m)
        sizes[repo] = tot
        print(f"== {repo}@{REPOS[repo][0][:8]}")
        for c, b in sorted(tot.items(), key=lambda x: -x[1]):
            print(f"  {c:38s} {b:15,d} B {b / GiB:8.3f} GiB")
        print(f"  {'TOTAL':38s} {sum(tot.values()):15,d} B {sum(tot.values()) / GiB:8.3f} GiB")

    print("\n# 2. parameter counts (measured from header shapes)")
    mtp = collections.Counter()
    for k, m in H["dbirks/Qwen3.8-27B-W4A16-AutoRound/model_extra_tensors.safetensors"].items():
        if k != "__metadata__": mtp[cat(k)] += math.prod(m["shape"])
    print("MTP module (bf16 source):", {k: f"{v:,}" for k, v in mtp.items()}, f"total {sum(mtp.values()):,}")
    for repo in ("incoai/Qwen3.8-27B-DFlash2", "syvai/Qwen3.8-27B-DFlash2-W4A16"):
        comp = collections.Counter(); q4 = 0
        for k, m in H[repo + "/model.safetensors"].items():
            if k == "__metadata__": continue
            p = params(m, k); comp[dflash_cat(k)] += p
            if k.endswith("weight_packed"): q4 += p
        print(f"{repo}: total {sum(comp.values()):,} params, int4 {q4:,}, bf16 {sum(comp.values()) - q4:,}")
        for c, p in sorted(comp.items(), key=lambda x: -x[1]):
            print(f"  {c:34s} {p:15,d}")

    print("\n# 3. int8 g128 symmetric heads written by prepare/ (estimated: q bytes + 16-bit scale per 128)")
    V, Hd = 248320, 5120
    def int8(rows, cols): return rows * cols + rows * (cols // 128) * 2
    lm8, emb8, dh8 = int8(V, Hd), int8(V, Hd), int8(40960, Hd)
    mtp_lin = [(12288, 5120), (1024, 5120), (1024, 5120), (5120, 6144), (17408, 5120), (17408, 5120), (5120, 17408)]
    mtp8 = sum(int8(r, c) for r, c in mtp_lin)
    fc8 = int8(5120, 10240)  # prepare/quant_mtp.py:36-37 quantizes mtp.fc unless --keep-fc
    print(f"  lm_head int8 {lm8:,} B ({lm8 / GiB:.3f} GiB); embed int8 {emb8 / GiB:.3f} GiB;"
          f" draft_lm_head int8 (40,960 rows) {dh8 / GiB:.3f} GiB")
    print(f"  MTP linears int8 {mtp8 / GiB:.3f} GiB + fc int8 {fc8 / GiB:.3f} GiB")

    print("\n# 4. resident weights per variant, language-model-only (estimated, sums of rows above)")
    b, f = sizes["dbirks/Qwen3.8-27B-W4A16-AutoRound"], sizes["syvai/qwen3.8-27b-3090-fast-variant"]
    body = b["body linears int4 (packed+scales)"] + b["gdn in_proj_a/b (bf16)"] + b["gdn conv1d/A_log/dt/norm"] + b["norms/other"]
    mtp_fast = f["mtp.linears"] + f["mtp.fc"] + f["mtp.draft_lm_head"] + f["mtp.norms"]
    mtp_base = mtp8 + fc8 + dh8
    dfl = sum(sizes["syvai/Qwen3.8-27B-DFlash2-W4A16"].values())
    rows = {
        "body (int4 linears + bf16 GDN/norms)": body,
        "base heads: lm_head int8 + embed int8": lm8 + emb8,
        "fast heads: lm_head int4 GPTQ + embed int8": f["lm_head"] + emb8,
        "MTP base (int8 linears+fc, int8 draft head)": mtp_base,
        "MTP fast (int4 GPTQ linears+fc, int4 draft head)": mtp_fast,
        "DFlash2 W4A16 file": dfl,
    }
    for k, v in rows.items(): print(f"  {k:52s} {v / GiB:7.3f} GiB")
    print(f"  A batch, base, no drafter               {(body + lm8 + emb8) / GiB:7.3f} GiB")
    print(f"  D SPEC=mtp, fast variant                {(body + f['lm_head'] + emb8 + mtp_fast) / GiB:7.3f} GiB")
    print(f"  D SPEC=mtp, base variant                {(body + lm8 + emb8 + mtp_base) / GiB:7.3f} GiB")
    print(f"  B/C/E SPEC=dflash2, fast target         {(body + f['lm_head'] + emb8 + dfl) / GiB:7.3f} GiB")
    print(f"  B with the bf16 DFlash2 instead         {(body + f['lm_head'] + emb8 + sum(sizes['incoai/Qwen3.8-27B-DFlash2'].values())) / GiB:7.3f} GiB")

    print("\n# 5. KV page geometry, launcher formula (start_qwen.sh:566-616); pool/block checked against"
          " bench/test_resident_print.sh:75-99")
    per_tok = {"bf16": (4096, 4096), "int8_per_token_head": (2080, 2112), "fp8": (2048, 2048)}
    for name, kv_mem, k, dt, expect in [
        ("B  CTX=fast bf16 k=7", 5583457484, 7, "bf16", (380, 448)),
        ("D' CTX=long int8 k=7 (FA2_FP8KV=0)", 5583457484, 7, "int8_per_token_head", (382, 864)),
        ("   CTX=long fp8 k=7 (FA2_FP8KV=1)", 5583457484, 7, "fp8", (387, 880)),
        ("C  CTX=fast bf16 k=15", 5583457484, 15, "bf16", None),
    ]:
        tgt, dft = per_tok[dt]
        state = 20480 * (3 + k) + 48 * 128 * 128 * 2
        block = 16 * -(-state // (16 * tgt))
        page = 8 * block * max(tgt, dft)
        pool = kv_mem // page
        st_pages = 6 * (2 + k)
        print(f"  {name:38s} block {block:4d} page {page / 2**20:6.2f} MiB pool {pool} pages"
              f" {'(no logged pool)' if expect is None else 'matches test' if expect == (pool, block) else 'MISMATCH ' + str(expect)};"
              f" state/request {st_pages} pages = {st_pages * page / GiB:.3f} GiB;"
              f" context {2 * page / block:,.0f} B/token (2 attention groups)")

    print("\n# 6. effective bytes per logged pool token = KV_MEM / pool (pins: start_qwen.sh:497,505,515,519;"
          " pools: long-context.md:125-126,190, README.md:102, long-context.md:571)")
    for name, kv_mem, pool in [("B bf16 k=7", 5583457484, 69758), ("C bf16 k=15", 5583457484, 57669),
                               ("D' int8 k=7, prefix cache", 5583457484, 136429),
                               ("dflash2 long fp8 FA2", 5583457484, 140902), ("E KVarN k=7", 5261334938, 268169)]:
        print(f"  {name:28s} {kv_mem / GiB:.2f} GiB / {pool:,} = {kv_mem / pool:9,.0f} B per pool token;"
              f" 1 GiB of KV_MEM = {GiB / (kv_mem / pool):8,.0f} tokens")
    print("  D EMBED_UVA A/B: 1.203 GiB freed (row 3) -> 204,336 to 236,479 tokens (single-user/README.md:443):"
          f" {emb8 / (236479 - 204336):,.0f} B per pool token")


if __name__ == "__main__":
    main()
