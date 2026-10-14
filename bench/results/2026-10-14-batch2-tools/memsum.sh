#!/bin/bash
# memsum.sh <dir> <label>: append the boot's memory evidence to <dir>/mem.txt
#   - engine memory lines from the boot log, verbatim
#   - nvidia-smi before/after boot and after each measurement (raw-smi-<label>.log)
#   - the 3090's adapter counters. The LUIDs changed between batch 1 (..114dd) and batch 2 (machine restart);
#     batch 2's 3090 instance is set in L below and its identification is in ../2026-10-14-batch2-preflight/.
# Spill rule: Shared Usage on this adapter is 0.2 MiB idle, and rises to 80-92 MiB when the engine creates its
# CUDA context, before the weights load (with >1.4 GiB dedicated still free). So the reference is the loaded
# engine's own floor: the minimum Shared Usage over samples with Dedicated > 20,000 MiB. A sample more than
# 16 MiB above that floor is flagged as a possible spill.
dir=$1; label=$2; L=${LUID:-luid_0x00000000_0x00011456_phys_0}
{
  echo "=== $label: engine memory lines (raw-boot-$label.log) ==="
  grep -aE 'Model loading took|Initial free memory|Available KV cache memory|GPU KV cache size|Graph capturing finished|Using .*attention backend|Using AttentionBackendEnum|kv_cache_dtype=|cudagraph_mode|downgrad' "$dir/raw-boot-$label.log" | grep -v 'vit attention\|MMEncoder' | sed 's/^single-1  | //' | cut -c1-330
  echo
  echo "=== $label: nvidia-smi, GPU 0 = 3090 (raw-smi-$label.log) ==="
  cat "$dir/raw-smi-$label.log"
  echo
  echo "=== $label: 3090 adapter counters ($L) from mem-trace-$label.csv ==="
  awk -F, -v L="$L" '
    $2==L {t=$1; if ($3=="dedicated usage") d[t]=$4/1048576; else s[t]=$4/1048576; ts[t]=1}
    END {
      floor=-1; maxs=0; maxd=0
      for (t in ts) { if (d[t]>20000 && s[t]>1 && (floor<0 || s[t]<floor)) floor=s[t]; if (s[t]>maxs) maxs=s[t]; if (d[t]>maxd) maxd=d[t] }
      printf "samples=%d  max dedicated=%.1f MiB  max shared=%.1f MiB  loaded-engine shared floor=%.1f MiB\n", length(ts), maxd, maxs, floor
      n=asorti(ts, k); c=0
      for (i=1;i<=n;i++) { t=k[i]; if (floor>=0 && d[t]>20000 && s[t]>floor+16) { printf "  POSSIBLE SPILL %s shared=%.1f MiB dedicated=%.1f MiB\n", t, s[t], d[t]; c++ } }
      if (!c) print "  no sample above floor+16 MiB: no spill seen"
    }' "$dir/mem-trace-$label.csv"
  echo
} >> "$dir/mem.txt"
tail -n 12 "$dir/mem.txt"
