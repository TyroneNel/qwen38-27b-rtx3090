# X-15: recompute the per-resident GDN state from the launcher's own formula
# (single-user/start_qwen.sh:608-614 @95ac251). Estimated: pure arithmetic, no boot.
import math
KV_MEM = 5583457484          # single-user/start_qwen.sh:519 (CTX=fast, dflash2)
GiB = 2**30
for name, k, tgt in (("DFlash2 k=7 bf16 (CTX=fast)", 7, 4096), ("DFlash2 k=15 bf16", 15, 4096),
                     ("k=4 bf16 (doc's 'MTP k=4')", 4, 4096), ("k=3 bf16 (MTP default)", 3, 4096)):
    state_b = 20480 * (3 + k) + 48 * 128 * 128 * 2     # :610 (fp16/bf16 SSM = 2 B)
    block = 16 * math.ceil(state_b / (16 * tgt))        # :611
    page = 8 * block * tgt                              # :612 (bf16: drafter width = target width)
    pool = KV_MEM // page                               # :613
    pages = 6 * (k + 2)                                 # :608,:614
    print(f"{name:30s} state/layer={state_b:,} B block={block} page={page:,} B pool={pool} "
          f"pages/resident={pages} -> {pages*page/GiB:.3f} GiB = {pages/(pool-1)*100:.1f}% of pool-1")
