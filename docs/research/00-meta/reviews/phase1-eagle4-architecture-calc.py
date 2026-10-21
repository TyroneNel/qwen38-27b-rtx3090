"""CPU-only recomputation of the memory/traffic estimates in 02-eagle4/architecture.md
and build-and-kernels.md. Inputs: HF safetensors header + config (EAGLE-Corp/Qwen3.8-27B-EAGLE4
@ f58a598), Qwen/Qwen3.8-27B config.json. All outputs are ESTIMATES (bandwidth floors)."""
GB, MiB, GiB = 1e9, 2**20, 2**30
BW = 936e9  # RTX 3090 peak DRAM bandwidth, B/s (spec sheet)

# draft tensors (params) from the safetensors header
H, V, R, I = 5120, 248320, 128, 16384
embed = lm_head = V * H
fc = 3 * H * H
factors = V * R + R * H
qkv = 3 * H * (2 * H)          # q/k/v [5120 out, 10240 in] each: input is cat(norm(embed), norm(hidden))
o = H * H
mlp = 3 * H * I
layer = qkv + o + mlp          # norms ignored (<0.1M)
total = embed + lm_head + fc + factors + layer
print(f"total params            {total/1e6:9.2f} M  (header: 3089.08 M incl. norms)")
print(f"decoder layer           {layer/1e6:9.2f} M = {layer*2/GB:.3f} GB bf16")
print(f"emb+lm_head share       {(embed+lm_head)/total*100:.1f} %")
print(f"resident if emb shared  {(total-embed)*2/GB:.3f} GB bf16")
print(f"resident if nothing shared {total*2/GB:.3f} GB bf16")

# KV per token
draft_kv = 2 * 40 * 128 * 2
target_kv = 16 * 2 * 4 * 256 * 2
print(f"draft KV/token {draft_kv} B, target KV/token {target_kv} B, ratio {draft_kv/target_kv*100:.2f} %")
win = 4096 * draft_kv
print(f"SWA window KV           {win/MiB:.0f} MiB = {win/GB:.3f} GB")

# per-step traffic
gather = 4 * 128 * H * 2       # 4 rows x 128 candidate rows of lm_head
eff = layer * 2 + factors * 2 + win + gather
full = layer * 2 + lm_head * 2 + win
print(f"step, efficient head    {eff/GB:.3f} GB  -> {eff/BW*1e3:.3f} ms")
print(f"step, full lm_head      {full/GB:.3f} GB  -> {full/BW*1e3:.3f} ms")
print(f"cut                     {(1-eff/full)*100:.1f} %")
prefill = layer * 2 + fc * 2 + factors * 2 + win
d6 = prefill + 5 * eff
print(f"depth-6 round           {d6/GB:.2f} GB -> {d6/BW*1e3:.2f} ms")
w4 = layer * 0.5 * (1 + 1/64) + factors * 2 + win  # int4 + g128 bf16 scales approx
print(f"W4A16 layer step        {w4/GB:.3f} GB -> {w4/BW*1e3:.2f} ms")

# our MTP drafter per step (start_qwen.sh:14-16: 215 MB module; int4 lm_head slice 40960 rows)
mtp_head = 40960 * H * 0.5 * (1 + 4/128)  # int4 + bf16 scale per g128 (approx)
print(f"MTP step                {(215e6 + mtp_head)/GB:.3f} GB (module 0.215 + head {mtp_head/GB:.3f})")

# GDN recurrent state per verify row
gdn = 48 * 48 * 128 * 128
print(f"GDN state/row fp32 {gdn*4/MiB:.0f} MiB, fp16 {gdn*2/MiB:.0f} MiB")
print(f"width-16 tree: fp32 {16*gdn*4/GiB:.2f} GiB, fp16 {16*gdn*2/GiB:.2f} GiB; "
      f"align-mode 2+k=17 slots fp16 {17*gdn*2/GiB:.2f} GiB")
