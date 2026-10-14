# Batch 3 Q1 supplementary: the Running: probe with the admission cap lifted

Not a Q1 arm. Q1 fixes MAX_SEQS=8, and on that probe k=7 already holds 7 of the 8 seats, so the 'rises by 3' test
cannot fire. These boots are the Q1 config with MAX_SEQS=16 and a 16-request wave (k=7, 3, 1), plus one abandoned
k=1 boot at MAX_SEQS=24 that spilled at the profiling step. Results, the spill and the read-out are in
../2026-10-14-Q1-k7/report.md (sections 'Supplementary' and 'Memory'). Raw: raw-running-<label>-w{1,2,3}.log,
raw-probe-*.json, raw-bootlines-*.log, mem.txt, cmd.txt, .env.<label>.
