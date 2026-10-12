# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch", "qwen-asr==0.0.6", "soundfile"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Step 2 (GPU notebook, or locally on CPU): force-align every chunk's audio to its exact book text with Qwen3-ForcedAligner.

usage: python align.py WORK_DIR [--batch N] [--dtype float32|float16|bfloat16] [--every N]
Uses CUDA when available (batched, float16 by default; with several GPUs one worker process each), otherwise CPU
(float32).
Writes timings.json with absolute per-sentence start/end, and units.json with the aligner's raw units per chunk
(times relative to the chunk start).
Resumable: finished chunks are kept in align-<dtype>/ (one cache per precision, so test runs never mix).
--every N aligns only every N-th chunk (a quick sample for tests; no timings file is written).
float16 vs float32 on a T4, whole book: identical sentence times except a few that moved by one 80 ms step; 3.5x faster.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import soundfile
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book import book_sentences, load_config, sentence_times, spoken_text, track_wav  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("work")
ap.add_argument("--batch", type=int, default=0, help="chunks per aligner call (default: 8 on GPU, 1 on CPU)")
ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                help="model precision (default: float16 on GPU, float32 on CPU)")
ap.add_argument("--every", type=int, default=1, help="align only every N-th chunk (test sample)")
ap.add_argument("--shard", help=argparse.SUPPRESS)       # "i/N": internal, one worker process per GPU
args = ap.parse_args()
cfg = load_config(os.path.abspath(args.work))
W = cfg["work"]
args.dtype = args.dtype or ("float16" if torch.cuda.is_available() else "float32")
if args.dtype != "float32" and not torch.cuda.is_available():
    sys.exit("half precision is only useful on a GPU")
A = f"{W}/align-{args.dtype}"                     # chunk cache for this precision
os.makedirs(A, exist_ok=True)
plan = json.load(open(f"{W}/plan.json"))
sentences = {s["id"]: s for s in book_sentences(cfg)}
chunks = plan["chunks"]
cuda = torch.cuda.is_available()
batch = args.batch or (8 if cuda else 1)


# cached chunk results are only valid for the plan they were made from
signature = hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest()
sig_file = f"{A}/plan.sha256"
if not os.path.exists(sig_file) or open(sig_file).read() != signature:
    for name in os.listdir(A):
        os.remove(f"{A}/{name}")
    open(sig_file, "w").write(signature)
todo = [n for n in range(0, len(chunks), args.every) if not os.path.exists(f"{A}/{n:04}.json")]

# Several GPUs (Kaggle's T4 x2): one worker process per GPU. Each track goes to one worker, longest first to the
# least loaded, so no track is decoded by two processes at once.
gpus = torch.cuda.device_count() if cuda else 0
if args.shard:
    i, n_shards = map(int, args.shard.split("/"))
    load_of, shard_of = [0.0] * n_shards, {}
    lengths = {}
    for c in chunks:
        lengths[c["track"]] = lengths.get(c["track"], 0.0) + c["end"] - c["start"]
    for t in sorted(lengths, key=lambda t: -lengths[t]):
        shard_of[t] = min(range(n_shards), key=lambda k: load_of[k])
        load_of[shard_of[t]] += lengths[t]
    todo = [n for n in todo if shard_of[chunks[n]["track"]] == i]
elif gpus > 1 and len(todo) > 1:
    import subprocess

    def relay(k, proc):
        for line in proc.stdout:
            print(f"GPU {k} | {line}", end="", flush=True)

    procs = []
    for k in range(gpus):
        cmd = [sys.executable, os.path.abspath(__file__), args.work, "--shard", f"{k}/{gpus}", "--dtype", args.dtype,
               "--every", str(args.every)] + (["--batch", str(args.batch)] if args.batch else [])
        proc = subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k)}, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        procs.append((proc, threading.Thread(target=relay, args=(k, proc))))
        procs[-1][1].start()
    t_all = time.time()
    for proc, thread in procs:
        proc.wait()
        thread.join()
    if any(proc.returncode for proc, _ in procs):
        sys.exit("a GPU worker failed - see its output above; run again to resume")
    total = sum(chunks[n]["end"] - chunks[n]["start"] for n in todo)
    print(f"{gpus} GPUs: {len(todo)} chunks in {(time.time() - t_all) / 60:.1f} min, "
          f"{total / (time.time() - t_all):.0f}x realtime overall", flush=True)
    todo = [n for n in todo if not os.path.exists(f"{A}/{n:04}.json")]
if todo:
    from qwen_asr import Qwen3ForcedAligner
    if not cuda:
        torch.set_num_threads(os.cpu_count())
    model = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=getattr(torch, args.dtype),
                                               device_map="cuda:0" if cuda else "cpu")
    print(f"aligning {len(todo)} chunks on {'GPU ' + torch.cuda.get_device_name(0) if cuda else 'CPU'}, "
          f"batch {batch}, {args.dtype}", flush=True)
    # Audio is loaded by background threads (decoding a track on first use) while the GPU aligns the current batch.
    track_locks = {t: threading.Lock() for t in {chunks[n]["track"] for n in todo}}

    def load(n):
        c = chunks[n]
        with track_locks[c["track"]]:            # one decode per track, even if two chunks ask at once
            wav = track_wav(cfg, c["track"])
        sr = soundfile.info(wav).samplerate
        audio, _ = soundfile.read(wav, start=int(c["start"] * sr), stop=int(c["end"] * sr), dtype="float32")
        return (audio, sr), "".join(spoken_text(sentences[i]["text"]) for i in c["sentences"])

    loader, loading = ThreadPoolExecutor(2), {}
    t0, done, finished = time.time(), 0.0, 0
    total = sum(chunks[n]["end"] - chunks[n]["start"] for n in todo)
    queue = list(todo)
    while queue:
        for n in queue[:2 * batch]:               # this batch and the next one
            if n not in loading:
                loading[n] = loader.submit(load, n)
        ns = queue[:batch]
        audios, texts = zip(*(loading[n].result() for n in ns))
        audios, texts = list(audios), list(texts)
        try:
            with torch.inference_mode():          # the library doesn't disable autograd itself
                results = model.align(audio=audios, text=texts, language=[cfg["language"]] * len(ns))
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
            batch = max(1, batch // 2)
            torch.cuda.empty_cache()
            print(f"out of GPU memory - retrying with batch {batch}", flush=True)
            continue
        for n, units in zip(ns, results):
            tmp = f"{A}/{n:04}.json.tmp"                  # written whole, then renamed: a resume never sees half a file
            raw = [[u.text, round(u.start_time, 3), round(u.end_time, 3)] for u in units]
            json.dump({"track": chunks[n]["track"], "sentences": sentence_times(chunks[n], raw, sentences),
                       "units": raw}, open(tmp, "w"), ensure_ascii=False)
            os.replace(tmp, f"{A}/{n:04}.json")
            done += chunks[n]["end"] - chunks[n]["start"]
        for n in ns:
            del loading[n]
        queue = queue[len(ns):]
        finished += len(ns)
        speed = done / (time.time() - t0)
        print(f"{finished}/{len(todo)} chunks | {speed:.0f}x realtime | ETA {(total - done) / speed / 60:.1f} min", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

if args.shard:
    sys.exit(0)                                   # a GPU worker: the parent process writes the timings
if args.every > 1:
    sys.exit(f"sample run: chunks are in {A}/ (no timings file written)")
timings, raw = {}, []
for n in range(len(chunks)):
    d = json.load(open(f"{A}/{n:04}.json"))
    for s in d["sentences"]:
        timings[s["id"]] = {**s, "track": d["track"]}
    raw.append({**chunks[n], "units": d.get("units")})
json.dump(timings, open(f"{W}/timings.json", "w"), ensure_ascii=False)
json.dump(raw, open(f"{W}/units.json", "w"), ensure_ascii=False)          # the aligner's own output, for diagnosis
print(f"timings.json ({args.dtype}):", sum(1 for t in timings.values() if t["start"] is not None), "sentences with times")
