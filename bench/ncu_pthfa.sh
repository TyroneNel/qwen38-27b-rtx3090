#!/usr/bin/env bash
# Nsight Compute on the pthfa verify kernel: tensor-pipe vs DRAM utilisation at 150k, T=8
# (docs/research/deepinfer/attention-kv.md open question 1 and attn-1 GPU validation step 3; LEDGER row attn-ncu),
# with T=1 at 150k and T=8 at 64k for comparison.
#
# Runs INSIDE the hq:exp-prof image (ncu at /usr/local/bin/ncu, pthfa in /app/venv), with this repo's bench/
# bind-mounted over /app/bench, the models mounted read-only (the attention shape comes from the target's
# config.json) and an output directory:
#
#   docker run --rm --gpus device=<UUID> --ipc host --cap-add SYS_ADMIN \
#     -v $REPO/bench:/app/bench -v /mnt/g/models:/app/models:ro -v $OUT:/out \
#     --entrypoint bash hq:exp-prof /app/bench/ncu_pthfa.sh [/out]
#
# Per config "T,CTX": bench/pthfa_verify_bench.py --kernel-only T,CTX launches the kernel ITERS times inside the
# NVTX push/pop range "pthfa_kernel" (after untimed warm-up launches outside it); ncu profiles only that range:
#   ncu --target-processes all --nvtx --nvtx-include pthfa_kernel/ --set full --metrics <KEY_METRICS> ...
# A launch with splits > 1 is two kernels (attention + combine); both are profiled. Writes, per config:
#   ncu_pthfa_T<T>_<CTX>.ncu-rep   the report (open in the Nsight Compute GUI)
#   ncu_pthfa_T<T>_<CTX>.raw.csv   every collected metric:  ncu --import ... --csv --page raw
#   ncu_pthfa_T<T>_<CTX>.key.csv   the key metrics only:    ncu --import ... --csv --page raw --metrics <KEY>
#   ncu_pthfa_T<T>_<CTX>.log       ncu + bench stdout/stderr
# and ncu_pthfa_summary.csv: one line per profiled kernel with the headline numbers.
#
# Knobs (environment):
#   CONFIGS="8,150000 1,150000 8,64000"  configs to profile
#   ITERS=3                              launches inside the NVTX range (each is profiled)
#   SET=full | focused                   full: --set full plus the key metrics; focused: the key metrics only
#                                        (a few replay passes instead of ~40; enough for the tensor-vs-DRAM question)
#   CLOCK_CONTROL=none | base            ncu's --clock-control. Default none: the GPU runs at its own (power-capped)
#                                        clocks, as when serving. base locks the SM at base clock (1395 MHz on a 3090),
#                                        which lowers the MMA rate but not DRAM bandwidth and so shifts the very
#                                        balance this measures; use it only for run-to-run comparability.
#   BENCH_ARGS=""                        extra pthfa_verify_bench.py args (e.g. --shape 24,4,256 --layers 4)
#   NCU=/usr/local/bin/ncu  PY=/app/venv/bin/python  BENCH=/app/bench/pthfa_verify_bench.py
#
# WSL2 / Windows: ncu needs access to the GPU performance counters. If the host denies it, ncu reports
# ERR_NVGPUCTRPERM; enable NVIDIA Control Panel > Desktop > Enable Developer Settings, then Developer >
# Manage GPU Performance Counters > "Allow access to the GPU performance counters to all users", and re-run.
#
# Exit: 0 = every config profiled; 2 = no measurement (no ncu / python / bench, counter permission denied,
# no kernel profiled, or ncu failed).
set -uo pipefail

OUT=${1:-/out}
CONFIGS=${CONFIGS:-"8,150000 1,150000 8,64000"}
ITERS=${ITERS:-3}
SET=${SET:-full}
CLOCK_CONTROL=${CLOCK_CONTROL:-none}
BENCH_ARGS=${BENCH_ARGS:-}
NCU=${NCU:-/usr/local/bin/ncu}
PY=${PY:-/app/venv/bin/python}
BENCH=${BENCH:-/app/bench/pthfa_verify_bench.py}
RANGE=pthfa_kernel

KEY_METRICS=(
  gpu__time_duration.sum
  sm__cycles_elapsed.avg.per_second
  dram__cycles_elapsed.avg.per_second
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
  sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active
  sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
  sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes_read.sum
  dram__bytes_read.sum.per_second
  lts__throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_sector_hit_rate.pct
  l1tex__throughput.avg.pct_of_peak_sustained_active
  smsp__issue_active.avg.pct_of_peak_sustained_active
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__maximum_warps_per_active_cycle_pct
  sm__cycles_active.avg
  sm__cycles_active.max
  sm__cycles_active.min
  launch__grid_size
  launch__block_size
  launch__registers_per_thread
  launch__occupancy_limit_registers
  launch__occupancy_limit_shared_mem
  "regex:smsp__average_warps_issue_stalled_.*_per_issue_active\.ratio"
)
KEY_LIST=$(IFS=,; echo "${KEY_METRICS[*]}")

fail() { echo "ncu_pthfa: NO RUN: $*" >&2; exit 2; }
[ -x "$NCU" ] || fail "no ncu at $NCU (run inside hq:exp-prof)"
[ -x "$PY" ] || fail "no python at $PY"
[ -f "$BENCH" ] || fail "no $BENCH (bind-mount the repo's bench/ over /app/bench)"
case "$SET" in full) SET_ARGS=(--set full) ;; focused) SET_ARGS=(--section LaunchStats) ;; *) fail "SET=$SET" ;; esac
mkdir -p "$OUT" || fail "cannot create $OUT"

"$NCU" --version | tail -1
nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu --format=csv,noheader 2>/dev/null || true

reports=()
for cfg in $CONFIGS; do
  T=${cfg%,*}; CTX=${cfg#*,}
  base="$OUT/ncu_pthfa_T${T}_${CTX}"
  echo "== ncu: T=$T ctx=$CTX -> $base.ncu-rep (set $SET, clock-control $CLOCK_CONTROL)"
  # shellcheck disable=SC2086
  "$NCU" --target-processes all --nvtx --nvtx-include "$RANGE/" "${SET_ARGS[@]}" --metrics "$KEY_LIST" \
    --clock-control "$CLOCK_CONTROL" --cache-control all --replay-mode kernel -f -o "$base" \
    "$PY" "$BENCH" --kernel-only "$T,$CTX" --iters "$ITERS" --nvtx-range "$RANGE" $BENCH_ARGS 2>&1 | tee "$base.log"
  rc=${PIPESTATUS[0]}
  if grep -q "ERR_NVGPUCTRPERM" "$base.log"; then
    fail "ERR_NVGPUCTRPERM: the host does not grant access to the GPU performance counters. On the Windows host:
  NVIDIA Control Panel > Desktop menu > Enable Developer Settings; then Developer > Manage GPU Performance Counters >
  'Allow access to the GPU performance counters to all users'. Apply, then re-run (no reboot needed on recent drivers)."
  fi
  grep -q "pthfa_verify_bench: NO RUN" "$base.log" && fail "the bench could not run (see $base.log)"
  grep -q "No kernels were profiled" "$base.log" && fail "no kernel was profiled in NVTX range '$RANGE/' (see $base.log)"
  [ "$rc" -eq 0 ] || fail "ncu exited $rc (see $base.log)"
  [ -f "$base.ncu-rep" ] || fail "no report written for T=$T ctx=$CTX"
  "$NCU" --import "$base.ncu-rep" --csv --page raw > "$base.raw.csv" || fail "ncu --import (raw) failed on $base.ncu-rep"
  "$NCU" --import "$base.ncu-rep" --csv --page raw --metrics "$KEY_LIST" > "$base.key.csv" \
    || fail "ncu --import (key metrics) failed on $base.ncu-rep"
  reports+=("$base.key.csv")
done

"$PY" - "$OUT/ncu_pthfa_summary.csv" "${reports[@]}" <<'EOF'
import csv, re, sys
out, paths = sys.argv[1], sys.argv[2:]
cols = [("time_us", "gpu__time_duration.sum"), ("sm_clk", "sm__cycles_elapsed.avg.per_second"),
        ("tensor_pipe_pct", "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"),
        ("tensor_inst_pct", "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active"),
        ("alu_pct", "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"),
        ("fma_pct", "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active"),
        ("dram_pct", "dram__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("dram_read_gbs", "dram__bytes_read.sum.per_second"),
        ("sm_pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("l2_pct", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        ("l1_pct", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
        ("occ_achieved_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
        ("occ_theory_pct", "sm__maximum_warps_per_active_cycle_pct"),
        ("sm_active_min", "sm__cycles_active.min"), ("sm_active_avg", "sm__cycles_active.avg"),
        ("sm_active_max", "sm__cycles_active.max"), ("grid", "launch__grid_size"),
        ("regs", "launch__registers_per_thread")]
STALL = re.compile(r"smsp__average_warps_issue_stalled_(.*)_per_issue_active\.ratio")
def num(s):
    try: return float(s.replace(",", ""))
    except ValueError: return None
def scale(unit, v):  # normalise time to us and rates to GHz / GB/s whatever unit ncu chose
    if v is None: return None
    u = unit.strip()
    f = {"nsecond": 1e-3, "usecond": 1, "msecond": 1e3, "second": 1e6,
         "hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1,
         "byte/second": 1e-9, "kbyte/second": 1e-6, "mbyte/second": 1e-3, "gbyte/second": 1, "tbyte/second": 1e3,
         "cycle/second": 1e-9, "cycle/nsecond": 1, "cycle/usecond": 1e-3, "cycle/msecond": 1e-6}.get(u.lower())
    return v * f if f is not None else v
rows = []
for p in paths:
    cfg = re.search(r"ncu_pthfa_(T\d+_\d+)\.key\.csv$", p).group(1)
    with open(p, newline="") as f:
        lines = [l for l in f if not l.startswith("==")]
    rd = list(csv.reader(lines))
    if len(rd) < 3:
        print(f"ncu_pthfa: {p}: no kernel rows", file=sys.stderr); continue
    head, units, data = rd[0], rd[1], rd[2:]
    idx = {h: i for i, h in enumerate(head)}
    for r in data:
        row = {"config": cfg, "kernel": r[idx["Kernel Name"]][:60] if "Kernel Name" in idx else ""}
        for name, m in cols:
            i = idx.get(m)
            v = scale(units[i], num(r[i])) if i is not None else None
            row[name] = "" if v is None else round(v, 3)
        stalls = sorted(((num(r[i]) or 0.0, STALL.match(h).group(1)) for h, i in idx.items() if STALL.match(h)),
                        reverse=True)
        row["top_stalls"] = " ".join(f"{n}={v:.2f}" for v, n in stalls[:4])
        rows.append(row)
if not rows:
    print("ncu_pthfa: no kernel rows in any report", file=sys.stderr); sys.exit(2)
fields = list(rows[0])
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fields); w.writeheader(); w.writerows(rows)
print(f"\n{'config':<12} {'kernel':<34} {'us':>8} {'GHz':>5} {'tens%':>6} {'dram%':>6} {'GB/s':>6} {'sm%':>5} "
      f"{'occ%':>5}  top stalls")
for r in rows:
    print(f"{r['config']:<12} {r['kernel'][:34]:<34} {r['time_us']!s:>8} {r['sm_clk']!s:>5} {r['tensor_pipe_pct']!s:>6} "
          f"{r['dram_pct']!s:>6} {r['dram_read_gbs']!s:>6} {r['sm_pct']!s:>5} {r['occ_achieved_pct']!s:>5}  {r['top_stalls']}")
print(f"wrote {out}")
EOF
rc=$?
[ "$rc" -eq 0 ] || exit 2
echo "ncu_pthfa: OK (${#reports[@]} configs) in $OUT"
