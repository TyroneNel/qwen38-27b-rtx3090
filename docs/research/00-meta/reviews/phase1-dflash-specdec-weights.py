"""Sum safetensors-header byte counts by tensor class for the target (base AutoRound shards +
fast-variant overrides) and the DFlash2 W4A16 drafter. Input: headers.json fetched by another
lane (HTTP range reads of safetensors headers). CPU-only, no weights downloaded."""
import json, re, sys, collections
H = json.load(open(sys.argv[1]))
def cls(name):
    if 'visual' in name: return 'vision'
    if name.startswith('mtp') or '.mtp.' in name: return 'mtp'
    if 'lm_head' in name: return 'lm_head'
    if 'embed_tokens' in name: return 'embed'
    if 'linear_attn' in name: return 'gdn_layers'
    if 'self_attn' in name: return 'attn_layers'
    if '.mlp.' in name: return 'mlp'
    if 'norm' in name: return 'norm'
    return 'other'
for shard, hdr in H.items():
    pass
tot = collections.defaultdict(lambda: collections.Counter())
names = collections.defaultdict(dict)
for shard, hdr in H.items():
    repo = shard.rsplit('/', 1)[0]
    for k, v in hdr.items():
        if k == '__metadata__': continue
        a, b = v['data_offsets']
        names[repo][k] = (b - a, shard)
for repo, d in names.items():
    c = collections.Counter()
    for k, (n, _) in d.items(): c[cls(k)] += n
    print(repo, {k: round(v / 1e9, 3) for k, v in sorted(c.items())}, 'total GB', round(sum(c.values()) / 1e9, 3))
# effective target: base shards 1-6 + the fast variant's shard 7 and extra tensors (want.json:
# the fast variant ships only those two files; the rest are the base shards).
eff = {k: v for k, v in names['dbirks/Qwen3.8-27B-W4A16-AutoRound'].items()
       if not v[1].endswith(('model-00007-of-00007.safetensors', 'model_extra_tensors.safetensors'))}
eff.update(names['syvai/qwen3.8-27b-3090-fast-variant'])
c = collections.Counter()
for k, (n, _) in eff.items(): c[cls(k)] += n
print('EFFECTIVE fast-variant', {k: round(v / 1e9, 3) for k, v in sorted(c.items())})
dec = sum(v for k, v in c.items() if k not in ('vision', 'mtp', 'embed'))
drafter = sum(n for n, _ in names['syvai/Qwen3.8-27B-DFlash2-W4A16'].values())
lm = c['lm_head']
print(f'target streamed per step (layers + norms + int4 lm_head; excl. vision, mtp, embed): {dec/1e9:.3f} GB')
print(f'drafter W4A16 file: {drafter/1e9:.3f} GB = {drafter/2**30:.3f} GiB')
for bw in (936.2, 0.85 * 936.2):
    print(f'@ {bw:.0f} GB/s  target floor {dec/1e9/bw*1e3:.2f} ms | drafter {drafter/1e9/bw*1e3:.2f} ms'
          f' | drafter+lm_head re-read {(drafter+lm)/1e9/bw*1e3:.2f} ms')
