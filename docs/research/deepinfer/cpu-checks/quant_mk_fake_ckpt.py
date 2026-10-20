import torch, sys
from safetensors.torch import save_file
sys.path.insert(0, ".")
exec(open("quant_b_signfold.py").read().split("R = 8192")[0])
torch.manual_seed(1)
w = torch.randn(256, 1024)
q, s = autoround_sym(w.view(256, 8, 128), torch.ones(256, 8), torch.ones(256, 8), torch.zeros(256, 8, 128))
u = (q.reshape(256, 1024) + 8).to(torch.int32).view(256, 128, 8)
packed = (u << torch.arange(0, 32, 4, dtype=torch.int32)).sum(-1, dtype=torch.int32)
import os; os.makedirs("fake_ckpt", exist_ok=True)
save_file({"model.layers.0.mlp.down_proj.weight_packed": packed.contiguous(), "model.layers.0.mlp.down_proj.weight_scale": s.half().contiguous()}, "fake_ckpt/model.safetensors")
