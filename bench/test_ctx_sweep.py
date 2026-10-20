#!/usr/bin/env python3
"""The context-profile sweep (bench/ctx_sweep) on CPU, with bash and no GPU.
Every arm in arms.txt runs through the real single-user launcher on a stub 8.6 GPU with the sweep's own stub vllm,
and analyze.py's dry-run check must pass on those launches. Its negative control, two arms that one launch
satisfies, must fail it. Then each memory and length step the plan takes must reach the launched argv, and
analyze.py's boot verdicts, plan and report run on fixture boots built from those launches.
  python3 bench/test_ctx_sweep.py
Exit 0 pass, 1 fail, 3 if the dry run's negative control passed.
"""
import contextlib, io, json, os, shutil, subprocess, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "bench", "ctx_sweep"))
import analyze as an  # noqa: E402

FAILS = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        FAILS.append(what)


def quiet(f, *a):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = f(*a)
    return rc, out.getvalue()


def stage(tree):
    subprocess.run(f"tar --exclude=.git --exclude=./venv --exclude=docs/media --exclude=bench/demo --exclude=./models "
                   f"-cf - . | tar -xf - -C {tree}", shell=True, cwd=REPO, check=True)
    for d in ("venv/bin", "smi", "models/Qwen3.8-27B-DFlash2-W4A16"):
        os.makedirs(os.path.join(tree, d), exist_ok=True)
    open(os.path.join(tree, "models/Qwen3.8-27B-DFlash2-W4A16/model.safetensors"), "w").close()
    shutil.copy(os.path.join(REPO, "bench/ctx_sweep/vllm_stub"), os.path.join(tree, "venv/bin/vllm"))
    for f, body in (("venv/bin/python", 'case "$*" in *pth_flash_attn*|*fa2_fp8kv*|*kvarn*) exit 0 ;; esac\nexec python3 "$@"\n'),
                    ("smi/nvidia-smi", "echo 8.6\n")):
        with open(os.path.join(tree, f), "w") as fh:
            fh.write("#!/bin/sh\n" + body)
        os.chmod(os.path.join(tree, f), 0o755)


def launch(tree, env):
    e = dict(PATH=f"{tree}/smi:/usr/bin:/bin", HOME=tree, HOST="127.0.0.1", **an.base(), **env)
    p = subprocess.run(["bash", "single-user/start_qwen.sh"], cwd=tree, env=e, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def labd(tag, decode, prompt, ttft):
    rows = [f"LABD {tag} {n:8s} prompt={prompt:6d} out= 512 steps=  150 tok/step= 3.41 decode={decode:6.1f} "
            f"e2e={decode / 2:6.1f} ttft={ttft:5.2f}s" for n in ("copy", "code", "edit", "quote", "summary", "qa")]
    return "\n".join(rows + [f"LABD {tag} TOTAL out=3072 steps=900 tok/step=3.41 decode={decode:.1f} tok/s"])


def cell_text(name, L, m, p):
    if name == "smoke":
        return f"SUMMARY {m.get('smoke', 12)} / 12 passed"
    if name.startswith("labd"):
        depth = int(name[4:])
        return labd(f"sw-{L}-{name}", m["decode"] * (1.1 if depth < 50 else 1.0), depth * 1000, depth / 10)
    if name == "ladder":
        c = m["decode"]
        return ("| N | 1 | 2 | 4 |\n|---|---|---|---|\n"
                f"| per-stream tok/s | {c:.1f} | {c * 0.8:.1f} | {c * 0.6:.1f} |\n| tok/step | 3.41 | 3.30 | 3.10 |\n"
                "| preemptions | 0 | 0 | 0 |")
    if name == "needle":
        return f"context ~200000 tokens, needle at 90% depth\nelapsed 40.0s, answer: 'x'\n{m.get('needle', 'RETRIEVED')}"
    gsm = f"\nsw-{L} GSM8K n=200 acc={m.get('gsm', 0.935):.3f} mean_tokens=210 90s" if p == 1 else ""
    return f"sw-{L} PPL {{'en': ({m.get('ppl', 10.76)}, 12000), 'da': (8.1, 11000), 'all': (9.4, 23000)}} 60s" + gsm


def make_boot(D, L, live, env, m):
    """One boot's files as sweep.sh boot leaves them, with numbers from m; the analyzer's record and verdict too."""
    B, run = os.path.join(D, L), an.envdict(open(os.path.join(D, "run.env")).read().splitlines())
    luid, p = run["LUID"], int(an.LABEL.fullmatch(L).group(2))
    write(f"{B}/env", "\n".join(f"{k}={v}" for k, v in {**an.base(), **env}.items()) + "\n")
    write(f"{B}/boot.status", m.get("status", "OK") + "\n")
    write(f"{B}/boot.log", m.get("bootlog", f"INFO GPU KV cache size: {m.get('pool', 180000):,} tokens\n"))
    write(f"{B}/image.txt", run["IMG_ID"] + "\n")
    write(f"{B}/smi.log", "# before boot\nmemory.total [MiB], memory.used [MiB], memory.free [MiB], power.limit [W]\n"
                          "24576 MiB, 120 MiB, 24200 MiB, 250.00 W, 20.00 W, 35, 210 MHz\n# after boot\n"
                          "24576 MiB, 23000 MiB, 1300 MiB, 250.00 W, 110.00 W, 50, 1695 MHz\n")
    rows = []
    for s in range(0, 600, 10):
        t = f"2026-10-20T12:{s // 60:02d}:{s % 60:02d}"
        ded = 500 if s < 30 else m.get("peak", 23000)
        sh = 100 + (m.get("rise", 0) if s >= 60 else 0)
        rows += [f"{t},{luid},dedicated usage,{ded * 1048576}", f"{t},{luid},shared usage,{sh * 1048576}",
                 f"{t},luid_0x0_0x1_phys_0,dedicated usage,{900 * 1048576}"]
    write(f"{B}/mem-trace-{L}.csv", "timestamp,instance,counter,bytes\n" + "\n".join(rows) + "\n")
    write(f"{B}/.kvline-{L}", "2026-10-20T12:01:00\n")
    write(f"{B}/.firstreq-{L}", "2026-10-20T12:03:00\n")
    write(f"{B}/restarts.txt", "after boot: 0 2026-10-20T11:59:00Z\nbefore down: 0 2026-10-20T11:59:00Z\n")
    write(f"{B}/serverlog.log", "INFO Application startup complete.\n")
    if m.get("status", "OK") == "OK":
        write(f"{B}/live.txt", live)
        write(f"{B}/gpu.txt", f"GPU 0: NVIDIA GeForce RTX 3090 (UUID: {an.base()['NVIDIA_VISIBLE_DEVICES']})\n{an.GPU_NAME}\n")
        r = m.get("r", 1.0)
        for name, _ in an.cells(D, L):
            gpu = "\n".join("GPU, 2026/10/20 12:05:00.000, P2, 1695 MHz, 9501 MHz, 240.50 W, 65, 98 %, 23000 MiB" for _ in range(3))
            write(f"{B}/cell-{name}.log", f"CLOCK 1000.0 500.0 1000.2\n# start x\n{cell_text(name, L, m, p)}\n{gpu}\n"
                                          f"# end x exit=0\nCLOCK {1000.1 + 100 * r} 600.0 {1000.3 + 100 * r}\n")
    if m.get("spill"):
        write(f"{B}/SPILL-{L}", "12:05:00 ded=23500 shared=300\n")
    rec, v = an.boot(D, L)
    write(f"{B}/analysis.txt", "RECORD " + json.dumps(rec) + "\n" + v + "\n")
    write(f"{B}/verdict", v + "\n")
    return v


def main():
    A = an.arms()
    check(len(A) >= 3 and {a["profile"] for a in A.values()} == {"fast", "long", "huge"}
          and all(a["expect"] for a in A.values()), f"arms.txt parses: {len(A)} arms over fast, long and huge, each with expectations")
    tmp = tempfile.mkdtemp(prefix="ctxsweep-")
    tree, D = os.path.join(tmp, "tree"), os.path.join(tmp, "D")
    os.makedirs(tree)
    stage(tree)
    write(f"{D}/run.env", "IMG=img:sha-x\nIMG_ID=sha256:abc\nLUID=luid_0x0_0x2_phys_0\nPASSES=2\n")
    outs = {}
    for name, a in A.items():
        rc, out, err = launch(tree, an.envdict(a["env"]))
        outs[name] = out
        write(f"{D}/dryrun/{name}.out", out)
        check(rc == 0 and an.launch(out)[0][:1] == ["serve"], f"{name}: the launcher starts the stub vllm (exit {rc}{', ' + err.strip().splitlines()[-1] if rc and err.strip() else ''})")
    rc, out = quiet(an.dryrun, D)
    check(rc == 0 and out.rstrip().endswith("DRYRUN_OK"), "every arm meets its expectations and each profile's arms are told apart"
          + ("" if rc == 0 else ":\n" + "\n".join(x for x in out.splitlines() if "FAIL" in x)))

    swapped = os.path.join(tmp, "swapped")
    shutil.copytree(D, swapped)
    write(f"{swapped}/dryrun/f-mtp3.out", outs["f-mtp4"])
    rc, out = quiet(an.dryrun, swapped)
    check(rc == 1 and "DRY f-mtp3 FAIL" in out, "f-mtp4's launch under f-mtp3's name fails f-mtp3's expectations")
    real = an.arms
    an.arms = lambda: {**real(), "f-twin": {**real()["f-mtp4"], "name": "f-twin"}}
    write(f"{swapped}/dryrun/f-twin.out", outs["f-mtp4"])
    write(f"{swapped}/dryrun/f-mtp3.out", outs["f-mtp3"])
    rc, out = quiet(an.dryrun, swapped)
    an.arms = real
    if not (rc == 1 and "DRY f-mtp4 and f-twin FAIL" in out):
        print("NEGATIVE CONTROL FAILED: two arms with one launch passed the dry run\n" + out)
        return 3
    check(True, "negative control: two arms that one launch satisfies fail the dry run")

    def step(arm, verdicts, p=1):
        for i, v in enumerate(verdicts, 1):
            write(f"{D}/{arm}-p{p}-{i}/verdict", v.replace("@", f"{arm}-p{p}-{i}") + "\n")
        plan = an.plan(D, arm, p)
        for i in range(1, len(verdicts) + 1):
            shutil.rmtree(f"{D}/{arm}-p{p}-{i}")
        return plan

    pin = an.flag(an.launch(outs["f-df7"])[0], "--kv-cache-memory")
    check(pin is not None, f"f-df7's launch pins --kv-cache-memory ({pin})")
    for arm, verdicts, key, want in (
            ("f-mtp4", ["BOOT_UNCLEAN mem @: SPILL"], "GPU_UTIL", "0.87"),
            ("f-mtp4", ["BOOT_UNCLEAN mem @: SPILL", "BOOT_UNCLEAN mem @: SPILL"], "GPU_UTIL", "0.85"),
            ("f-df7", ["BOOT_UNCLEAN mem @: SPILL"], "KV_MEM", str(int(pin or 0) - (256 << 20))),
            ("h-kvarn", ["RETRY_MAX_LEN 200704 @: the KV pool holds about 201234 tokens"], "MAX_LEN", "200704")):
        plan = step(arm, verdicts)
        env = an.envdict(plan[1:])
        rc, out, _ = launch(tree, env)
        argv, envl = an.launch(out)
        bad = an.check(an.expectations(A[arm], env), argv, envl, an.base()["MODEL"])
        check(plan[0] == f"BOOT {arm}-p1-{len(verdicts) + 1} {A[arm]['profile']}" and env.get(key) == want and rc == 0
              and an.flag(argv, an.FIT[key]) == want and not bad,
              f"{arm} after {len(verdicts)} x {verdicts[0].split()[0]}: {key}={want} reaches the argv as {an.FIT[key]} {want} {bad or ''}")

    for verdicts, p, want in (
            ([], 1, "BOOT f-mtp4-p1-1 fast"),
            (["BOOT_CLEAN @"], 1, "DONE f-mtp4 pass 1"),
            (["BOOT_UNCLEAN other @: smoke did not finish"], 1, "BOOT f-mtp4-p1-2 fast"),
            (["BOOT_UNCLEAN other @: x", "BOOT_UNCLEAN other @: x"], 1, "BROKEN f-mtp4 pass 1: two boots"),
            (["BOOT_UNCLEAN mem @: x"] * 3, 1, "BROKEN f-mtp4 pass 1: still unclean on memory"),
            (["BOOT_SETUP @: wrong image"], 1, "STOP f-mtp4-p1-1"),
            (["BOOT_UNCLEAN other @: x"] * 2, 2, "BROKEN f-mtp4 pass 2: no clean boot in an earlier pass")):
        plan = step("f-mtp4", verdicts, 1)
        if p == 2:
            for i, v in enumerate(verdicts, 1):
                write(f"{D}/f-mtp4-p1-{i}/verdict", v.replace("@", f"f-mtp4-p1-{i}") + "\n")
            plan = an.plan(D, "f-mtp4", 2)
            for i in range(1, len(verdicts) + 1):
                shutil.rmtree(f"{D}/f-mtp4-p1-{i}")
        check(plan[0].startswith(want), f"plan {verdicts or 'no boots'} in pass {p}: {plan[0]}")
    os.makedirs(f"{D}/f-mtp4-p1-1-aborted")
    plan = an.plan(D, "f-mtp4", 1)
    shutil.rmtree(f"{D}/f-mtp4-p1-1-aborted")
    check(plan[0] == "BOOT f-mtp4-p1-2 fast", f"an aborted boot keeps its label number: {plan[0]}")
    write(f"{D}/f-mtp4-p1-1/verdict", "BOOT_UNCLEAN mem f-mtp4-p1-1: x\n")
    write(f"{D}/f-mtp4-p1-2/verdict", "BOOT_CLEAN f-mtp4-p1-2\n")
    plan = an.plan(D, "f-mtp4", 2)
    for n in (1, 2):
        shutil.rmtree(f"{D}/f-mtp4-p1-{n}")
    check(plan[0] == "BOOT f-mtp4-p2-1 fast" and "GPU_UTIL=0.87" in plan, f"pass 2 starts at pass 1's fit: {plan}")

    mtp4env = an.envdict(A["f-mtp4"]["env"])
    for L, live, m, want in (
            ("f-mtp4-p1-1", outs["f-mtp4"], {"decode": 80}, "BOOT_CLEAN f-mtp4-p1-1"),
            ("f-mtp4-p1-2", outs["f-mtp4"], {"decode": 80, "spill": 1}, "BOOT_UNCLEAN mem f-mtp4-p1-2: SPILL"),
            ("f-mtp4-p1-3", outs["f-mtp4"], {"decode": 80, "rise": 40}, "BOOT_UNCLEAN mem f-mtp4-p1-3: boot rise 40.0"),
            ("f-mtp4-p1-4", outs["f-mtp4"], {"decode": 80, "peak": 23500}, "BOOT_UNCLEAN mem f-mtp4-p1-4: peak Dedicated 23500"),
            ("f-mtp4-p1-5", outs["f-mtp3"], {"decode": 80}, "BOOT_SETUP f-mtp4-p1-5: live argv lacks \"num_speculative_tokens\":4"),
            ("f-mtp4-p1-6", outs["f-mtp4"], {"decode": 80, "peak": 15000}, "BOOT_SETUP f-mtp4-p1-6: no 3090 counter sample"),
            ("f-mtp4-p1-7", "", {"status": "FAILED container state: 1 restarting",
                                 "bootlog": "ValueError: ... the estimated maximum model length is 201234.\n"},
             "RETRY_MAX_LEN 200704 f-mtp4-p1-7")):
        v = make_boot(D, L, live, mtp4env, m)
        check(v.startswith(want), f"boot fixture {L}: {v}")
    for n in range(1, 8):
        shutil.rmtree(f"{D}/f-mtp4-p1-{n}")

    def boots(arm, m, env=None, live=None):
        for p in (1, 2):
            v = make_boot(D, f"{arm}-p{p}-1", live or outs[arm], env or an.envdict(A[arm]["env"]), m)
            if not v.startswith("BOOT_CLEAN"):
                check(False, f"report fixture {arm}-p{p}-1 is clean: {v}")
    boots("f-mtp4", {"decode": 100})
    boots("f-mtp3", {"decode": 105, "r": 1.12})
    boots("f-df7", {"decode": 140, "smoke": 11})
    boots("f-df15", {"decode": 150, "ppl": 11.2})
    boots("l-fp8", {"decode": 60, "pool": 100000})
    boots("l-pth", {"decode": 50, "pool": 200000})
    boots("l-df", {"decode": 70, "pool": 210000, "needle": "MISSED"})
    boots("h-kvarn", {"decode": 30, "pool": 300000})
    h = {**an.envdict(A["h-int4"]["env"]), "MAX_LEN": "200704"}
    boots("h-int4", {"decode": 40, "pool": 205000}, h, launch(tree, h)[1])
    rc, rep = quiet(an.report, D)
    rows = {}
    for line in rep.splitlines():
        if line.startswith("| "):
            rows.setdefault(line.split(" | ")[0].strip("| "), line)
    check("| fast | f-mtp4 | CLEAR over f-mtp3 |" in rep,
          "fast: f-mtp4 wins once f-mtp3's 5% higher VM-clock numbers are put on the host clock (r 1.12)")
    check("no: smoke 11/12" in rows.get("f-df7", "") and "no: PPL en 11.2 above 1.03" in rows.get("f-df15", ""),
          "fast: the faster arms fail the smoke and the PPL bars")
    check("| long | l-pth | CLEAR over l-fp8 |" in rep and "no: needle MISSED" in rows.get("l-df", ""),
          "long: the larger pool wins on speed x context, and a missed needle is ineligible")
    check("| huge | h-kvarn | CLEAR over h-int4 |" in rep, "huge: the longer MAX_LEN wins over the faster decode")
    check("MAX_LEN=262144" in rep.split("## fast")[0] and "| f-mtp3-p2-1 | BOOT_CLEAN f-mtp3-p2-1 |" in rep,
          "the report gives the winner's env and lists every boot")
    shutil.rmtree(tmp)

    print("FAIL" if FAILS else "PASS", f"{len(FAILS)} failed" if FAILS else "")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
