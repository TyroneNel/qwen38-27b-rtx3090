"""Unit tests (synthetic logprob streams) and one end-to-end test through mock_server_x08.py. No GPU.
  python -m pytest -q test_classify_divergence.py      or      python test_classify_divergence.py
Synthetic means made up here: nothing below is a measurement of the real stack."""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

import classify_divergence as C

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def run(ids, gaps, alts=None, top=True):
    """A run: token ids, the top1-top2 gap at every position (None = no logprobs), runner-up ids (default id+1)."""
    alts = alts or [t + 1 for t in ids]
    r = dict(ids=list(ids), margins=list(gaps), finish="stop")
    if top:
        r["top"] = [[[t, -0.1], [al, -0.1 - g]] if g is not None else [] for t, g, al in zip(ids, gaps, alts)]
    return r


def flip(ids, gaps, k, gap_a=None, gap_b=None):
    """Reference run and a candidate that takes the reference's runner-up at k (a one-token flip, then it differs)."""
    a = run(ids, gaps)
    ib, gb = list(ids), list(gaps)
    ib[k] = ids[k] + 1
    gaps_a = list(gaps); gaps_a[k] = gap_a if gap_a is not None else gaps[k]
    gb[k] = gap_b if gap_b is not None else gaps[k]
    a = run(ids, gaps_a)
    b = run(ib, gb, alts=[ids[k] if i == k else t + 1 for i, t in enumerate(ib)])
    return a, b


IDS = list(range(100, 120))
BIG = [2.0] * 20


def test_identical():
    e = C.classify_pair(run(IDS, BIG), run(IDS, BIG))
    assert e["status"] == "identical"


def test_benign_near_tie_flip():
    a, b = flip(IDS, BIG, 7, 0.004, 0.004)
    e = C.classify_pair(a, b)
    assert e["status"] == "near_tie" and e["k"] == 7 and not e["notes"], e


def test_real_divergence_large_margin():
    a, b = flip(IDS, BIG, 7)
    e = C.classify_pair(a, b)
    assert e["status"] == "real" and e["gap"] == 2.0, e


def test_divergence_at_token0():
    for gap, want in ((0.002, "near_tie"), (3.0, "real")):
        a, b = flip(IDS, BIG, 0, gap, gap)
        e = C.classify_pair(a, b)
        assert e["status"] == want and e["k"] == 0 and "token0" in e["notes"], e


def test_divergence_at_last_token():
    for gap, want in ((0.002, "near_tie"), (3.0, "real")):
        a, b = flip(IDS, BIG, 19, gap, gap)
        e = C.classify_pair(a, b)
        assert e["status"] == want and e["k"] == 19 and "last_token" in e["notes"], e


def test_missing_logprobs():
    gaps = list(BIG); gaps[7] = None
    a, b = flip(IDS, BIG, 7, 0.001, 0.001)
    # both runs lack logprobs at k: fail closed
    a["top"][7], b["top"][7], a["margins"][7], b["margins"][7] = [], [], None, None
    assert C.classify_pair(a, b)["status"] == "unknown"
    # no logprobs at all in either file (record without logprobs)
    assert C.classify_pair(run(IDS, BIG, top=False) | dict(margins=[]), flip(IDS, BIG, 7)[1] | dict(top=[], margins=[]))["status"] == "unknown"
    # one side only: a tiny gap is never enough for near_tie
    a, b = flip(IDS, BIG, 7, 0.001, 0.001)
    b["top"][7], b["margins"][7] = [], None
    e = C.classify_pair(a, b)
    assert e["status"] == "ulp_band" and "one_sided" in e["notes"], e
    # one side only and large: still real
    a, b = flip(IDS, BIG, 7)
    b["top"][7], b["margins"][7] = [], None
    assert C.classify_pair(a, b)["status"] == "real"


def test_margins_only_legacy_format():  # greedy_equal.py record output: no "top"
    a, b = flip(IDS, BIG, 7, 0.004, 0.004)
    del a["top"], b["top"]
    assert C.classify_pair(a, b)["status"] == "near_tie"
    a, b = flip(IDS, BIG, 7)
    del a["top"], b["top"]
    assert C.classify_pair(a, b)["status"] == "real"


def test_ulp_band_needs_replay_and_judge():
    a, b = flip(IDS, BIG, 7, 0.1, 0.1)  # between 1e-2 and one bf16 ulp
    e = C.classify_pair(a, b)
    assert e["status"] == "ulp_band" and e["needs_rescore"]
    ref, cand = e["ref_token"], e["cand_token"]
    assert C.judge_one(e, dict(top=[[ref, -0.1], [cand, -0.2]]))["status"] == "replay_tie"      # d 0.1 <= 0.25
    assert C.judge_one(e, dict(top=[[cand, -0.1], [ref, -1.9]]))["status"] == "replay_argmax"   # cand wins at replay
    assert C.judge_one(e, dict(top=[[ref, -0.1], [cand, -1.5]]))["status"] == "real"            # d 1.4
    assert C.judge_one(e, dict(top=[[ref, -0.1], [cand + 5, -0.2], [cand + 6, -2.0]]))["status"] == "real"  # cand >=1.9 below
    assert C.judge_one(e, dict(top=[[ref, -0.1], [cand + 5, -0.12], [cand + 6, -0.15]]))["status"] == "unknown"
    assert C.judge_one(e, dict(top=[]))["status"] == "unknown"
    assert C.exit_code({"ulp_band": 1, "near_tie": 3}) == 3 and C.exit_code({"replay_tie": 1}) == 0


def test_candidate_not_runner_up_is_not_near_tie():
    a, b = flip(IDS, BIG, 7, 0.002, 0.002)
    b["ids"][7] = 999  # a third token: the saved top-2 cannot show the gap
    b["top"][7] = [[999, -0.1], [IDS[7], -0.102]]  # cand sees the ref token as runner-up; ref top-2 lacks 999
    e = C.classify_pair(a, b)
    assert e["status"] == "ulp_band", e  # lower bound only


def test_thresholds_configurable():
    a, b = flip(IDS, BIG, 7, 0.05, 0.05)
    assert C.classify_pair(a, b)["status"] == "ulp_band"
    assert C.classify_pair(a, b, near_tie=0.1)["status"] == "near_tie"
    assert C.classify_pair(a, b, ulp_band=0.04)["status"] == "real"


def test_length_divergence():
    a = run(IDS, BIG); b = run(IDS[:12], BIG[:12])  # b stopped early; a continued with margin 2.0
    e = C.classify_pair(a, b)
    assert e["status"] == "real" and e["k"] == 12 and "length_divergence" in e["notes"], e
    g = list(BIG); g[12] = 0.003
    e = C.classify_pair(run(IDS, g), b)
    assert e["status"] == "ulp_band" and e["needs_rescore"]  # EOS is not in ids: never a plain near_tie
    e["ref_token"], e["cand_token"] = IDS[12], None
    assert C.judge_one(e, dict(top=[[IDS[12], -0.1], [7, -0.11]]))["status"] == "replay_tie"


def test_kvarn_adaptive_prefix_shape():
    """Synthetic stand-in for the known case (docs/research/01-project/dflash-specdec.md:470-473): the cold pass is clean, the
    cache-hit repeat of a copy prompt goes wrong ~10 tokens in, with a confident reference. NOT measured here."""
    cold = run(IDS, BIG)
    ref = {"label": "off", "max_tokens": 20, "items": {"long/copy": cold, "long/copy@p2": cold}}
    bad = run(IDS[:10] + [555, 556] + IDS[12:], BIG)  # diverges at k=10 and stays wrong
    bad["top"][10] = [[555, -0.05], [556, -4.0]]
    cand = {"label": "adaptive", "max_tokens": 20, "items": {"long/copy": cold, "long/copy@p2": bad}}
    pinned = {"label": "pinned", "max_tokens": 20, "items": {"long/copy": cold, "long/copy@p2": cold}}
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for n, r in (("off", ref), ("adaptive", cand), ("pinned", pinned)):
            paths.append(os.path.join(d, n + ".json")); json.dump(r, open(paths[-1], "w"))
        out = os.path.join(d, "rep.json")
        assert C.main(["classify", paths[0], paths[2], "--out", out]) == 0
        assert C.main(["classify", paths[0], paths[1], "--out", out]) == 1
        e = json.load(open(out))["adaptive"]["entries"]
        assert e["long/copy"]["status"] == "identical" and e["long/copy@p2"]["status"] == "real" and e["long/copy@p2"]["k"] == 10
        assert C.main(["classify", paths[0], paths[2], paths[1]]) == 1  # three-way: one clean, one bad


def test_invalid_input():
    with tempfile.TemporaryDirectory() as d:
        x = os.path.join(d, "x.json"); y = os.path.join(d, "y.json")
        json.dump({"label": "x", "max_tokens": 4, "items": {"a": run(IDS, BIG)}}, open(x, "w"))
        json.dump({"label": "y", "max_tokens": 4, "items": {"b": run(IDS, BIG)}}, open(y, "w"))
        assert C.main(["classify", x, y]) == 2


# ---------- end to end through the mock copy: record_top2 -> classify -> rescore_replay -> judge ----------
def mock(port, *flags):
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "mock_server_x08.py"), "--port", str(port), "--rate", "5000", *flags])
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health"); return p
        except OSError:
            time.sleep(0.1)
    p.kill(); raise RuntimeError("mock did not start")


def sh(*cmd, port, check=None):
    env = dict(os.environ, PORT=str(port), OPENAI_API_KEY="k", PYTHONPATH=os.path.join(REPO, "bench"))
    env.pop("VLLM_API", None); env.pop("VLLM_MODEL", None)
    r = subprocess.run([sys.executable, *cmd], env=env, capture_output=True, text=True)
    if check is not None:
        assert r.returncode == check, (cmd, r.returncode, r.stdout[-800:], r.stderr[-800:])
    return r


def record(label, d, port, *flags):
    p = mock(port, *flags)
    try:
        sh(os.path.join(HERE, "record_top2.py"), label, "--max-tokens", "24", "--passes", "2", "--out", d, port=port, check=0)
    finally:
        p.kill(); p.wait()


def test_end_to_end_mock():
    with tempfile.TemporaryDirectory() as d:
        ref = os.path.join(d, "greedy_off.json")
        record("off", d, 18971, "--tie-at", "5")
        it = json.load(open(ref))["items"]
        assert "short_chat/sc0@p2" in it and len(it["short_chat/sc0"]["top"][0]) == 5 and it["short_chat/sc0"]["messages"]
        for name, flags, want_status, want_exit in (
            ("tie", ("--tie-at", "5", "--perturb", "5"), "near_tie", 0),          # tie margin 0.005
            ("same", ("--tie-at", "5"), "identical", 0),
            ("real", ("--perturb-from", "7"), "real", 1),
        ):
            record(name, d, 18972, *flags)
            rep = os.path.join(d, f"rep_{name}.json")
            r = sh(os.path.join(HERE, "classify_divergence.py"), "classify", ref, os.path.join(d, f"greedy_{name}.json"),
                   "--out", rep, port=0, check=want_exit)
            st = {e["status"] for e in json.load(open(rep))[name]["entries"].values()}
            assert st == {want_status}, (name, st, r.stdout[-600:])
        # ulp band: tie margin 0.1 -> ulp_band (exit 3) -> replay against a clean SPEC=off mock -> replay_tie (exit 0)
        record("off_band", d, 18973, "--tie-at", "5", "--tie-margin", "0.1")
        record("band", d, 18974, "--tie-at", "5", "--tie-margin", "0.1", "--perturb", "5")
        rep, req, rp = (os.path.join(d, n) for n in ("rep_band.json", "req.json", "rp.json"))
        sh(os.path.join(HERE, "classify_divergence.py"), "classify", os.path.join(d, "greedy_off_band.json"),
           os.path.join(d, "greedy_band.json"), "--out", rep, "--rescore-out", req, port=0, check=3)
        n_req = len(json.load(open(req)))
        assert n_req == 16 * 2, n_req
        p = mock(18975, "--tie-at", "5", "--tie-margin", "0.1")
        try:
            sh(os.path.join(HERE, "rescore_replay.py"), req, "--out", rp, port=18975, check=0)
        finally:
            p.kill(); p.wait()
        sh(os.path.join(HERE, "classify_divergence.py"), "judge", rep, rp, "--out", os.path.join(d, "final.json"), port=0, check=0)
        st = {e["status"] for e in json.load(open(os.path.join(d, "final.json")))["band"]["entries"].values()}
        assert st == {"replay_tie"}, st


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        f(); print("ok  ", n)
    print(f"ALL {len(tests)} PASS")
