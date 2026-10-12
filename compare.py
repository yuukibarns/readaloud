# /// script
# requires-python = "==3.12.*"
# dependencies = []
# ///
"""Compare sentence timings of two alignment runs on the chunks both have aligned (e.g. float32 vs float16).

usage: python compare.py WORK_DIR [REF_DIR TEST_DIR]      (default: align-float32 vs align-float16)
"""
import glob
import json
import os
import sys

W = sys.argv[1]
ref_name, other = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("align-float32", "align-float16")
ref_dir, test_dir = os.path.join(W, ref_name), os.path.join(W, other)
d_start, d_end, lost, gained, n_chunks = [], [], 0, 0, 0
for f in sorted(glob.glob(f"{test_dir}/[0-9]*.json")):
    ref_f = os.path.join(ref_dir, os.path.basename(f))
    if not os.path.exists(ref_f):
        continue
    ref = {s["id"]: s for s in json.load(open(ref_f))["sentences"]}
    n_chunks += 1
    for s in json.load(open(f))["sentences"]:
        r = ref[s["id"]]
        if r["start"] is None or s["start"] is None:
            lost += r["start"] is not None
            gained += s["start"] is not None
            continue
        d_start.append(abs(s["start"] - r["start"]))
        d_end.append(abs(s["end"] - r["end"]))
if not d_start:
    sys.exit("no chunks aligned in both runs")


def stats(name, d):
    d = sorted(d)
    q = lambda p: d[min(len(d) - 1, int(p * len(d)))]
    within = lambda t: sum(x <= t for x in d) / len(d)
    print(f"{name:6} median {q(.5):.3f}s  p95 {q(.95):.3f}s  max {d[-1]:.3f}s  "
          f"within 0.05s {within(.05):.1%}  within 0.15s {within(.15):.1%}")


print(f"{n_chunks} chunks, {len(d_start)} sentences compared ({other} vs {ref_name})")
stats("start", d_start)
stats("end", d_end)
print(f"sentences timed only in {ref_name}: {lost}; only in {other}: {gained}")
