import sys, math, torch
sys.path.insert(0, ".")
exec(open("quant_a_act_quant.py").read().split("T, N = 256, 1024")[0])
torch.manual_seed(0)
T, N, K = 256, 1024, 5120
H128 = hadamard(128)
for label, x in [("gaussian (bench/test_marlin_int8_asym-like)", torch.randn(T, K)),
                 ("student-t df=8, lognormal ch", make_acts(T, K, 8.0, 0, 1.0, 2))]:
    W = torch.randn(K, N) / math.sqrt(K); y = x @ W
    Wr = (W.view(K // 128, 128, N).transpose(1, 2) @ H128).transpose(1, 2).reshape(K, N)
    print(label, "per-token %.3f%%  per-128 %.3f%%  had128+per-token %.3f%%" % (
        100*rel(int8_per_token(x) @ W, y), 100*rel(int8_group(x, 128) @ W, y),
        100*rel(int8_per_token(had_block(x, 128, H128)) @ Wr, y)))
