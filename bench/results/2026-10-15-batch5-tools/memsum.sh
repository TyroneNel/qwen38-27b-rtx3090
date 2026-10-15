#!/bin/bash
# memsum.sh <dir> <label>: append the boot's memory evidence to <dir>/mem.txt (batch 3's, with batch 4's floors)
#   - engine memory lines from the boot log, verbatim
#   - nvidia-smi before/after boot and after each measurement (raw-smi-<label>.log)
#   - the 3090's adapter counters, and the floors spillwatch.sh used: F_ctx (before the KV-cache line, the CUDA-context
#     level) and F_kv (KV-cache line -> first request, the brief's floor). A boot spill is F_kv > F_ctx + 8 MiB; a
#     measurement spill is a sample after the first request above F_kv + 16 MiB (the guard needs two in a row).
dir=$1; label=$2; L=${LUID:-luid_0x00000000_0x00011456_phys_0}
KV=$(cat "$dir/.kvline-$label" 2>/dev/null); FR=$(cat "$dir/.firstreq-$label" 2>/dev/null)
{
  echo "=== $label: engine memory lines (raw-boot-$label.log) ==="
  grep -aE 'Model loading took|Initial free memory|Available KV cache memory|GPU KV cache size|Graph capturing finished|Using .*attention backend|Using AttentionBackendEnum|kv_cache_dtype=|cudagraph_mode|downgrad' "$dir/raw-boot-$label.log" | grep -v 'vit attention\|MMEncoder' | sed 's/^single-1  | //' | cut -c1-330
  echo
  echo "=== $label: nvidia-smi, GPU 0 = 3090 (raw-smi-$label.log) ==="
  cat "$dir/raw-smi-$label.log"
  echo
  echo "=== $label: 3090 adapter counters ($L) from mem-trace-$label.csv; KV-cache line $KV, first request $FR (local) ==="
  awk -F, -v L="$L" -v KV="$KV" -v FR="$FR" '
    $2==L {t=$1; if ($3=="dedicated usage") d[t]=$4/1048576; else s[t]=$4/1048576; ts[t]=1}
    END {
      fc=-1; fk=-1; maxs=0; maxd=0; n=asorti(ts, k)
      for (i=1;i<=n;i++) { t=k[i]; if (s[t]>maxs) maxs=s[t]; if (d[t]>maxd) maxd=d[t]; if (!(s[t]>1)) continue
        if (t<KV) { if (d[t]>1000 && (fc<0 || s[t]<fc)) fc=s[t] } else if (t<FR) { if (fk<0 || s[t]<fk) fk=s[t] } }
      printf "samples=%d  max dedicated=%.1f MiB  max shared=%.1f MiB  F_ctx=%.1f MiB  F_kv=%.1f MiB (floor used)\n", n, maxd, maxs, fc, fk
      if (fc>=0 && fk>fc+8) printf "  BOOT RISE: F_kv is %.1f MiB above F_ctx (MAX_SEQS=8 boots: 8.0)\n", fk-fc; else print "  boot: F_kv within 8 MiB of F_ctx (no boot rise beyond a MAX_SEQS=8 boot)"
      c=0
      for (i=1;i<=n;i++) { t=k[i]; if (fk>=0 && t>=FR && d[t]>20000 && s[t]>fk+16) { printf "  POSSIBLE SPILL %s shared=%.1f MiB dedicated=%.1f MiB\n", t, s[t], d[t]; c++ } }
      if (!c) print "  measurements: no sample after the first request above F_kv+16 MiB: no spill seen"
      printf "  trace (time shared dedicated), every 6th sample:\n"
      for (i=1;i<=n;i+=6) { t=k[i]; printf "    %s %.1f %.1f\n", t, s[t], d[t] }
    }' "$dir/mem-trace-$label.csv"
  echo
} >> "$dir/mem.txt"
tail -n 8 "$dir/mem.txt"
