# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch", "silero-vad", "soundfile"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Step 1b (local, called by prepare.py): speech segments in every track with Silero VAD; chunks get cut in the pauses.

usage: python vad.py WORK_DIR      (one process per CPU core; the model is tiny, so this is CPU work)
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book import load_config, track_wav  # noqa: E402

cfg = load_config(os.path.abspath(sys.argv[1]))


def run(track_id):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # torch.jit.load deprecation inside silero-vad
    import soundfile
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    torch.set_num_threads(1)
    wav = torch.from_numpy(soundfile.read(track_wav(cfg, track_id), dtype="float32")[0])
    segs = get_speech_timestamps(wav, load_silero_vad(), sampling_rate=16000, return_seconds=True,
                                 min_silence_duration_ms=120, speech_pad_ms=0)
    return track_id, [[s["start"], s["end"]] for s in segs]


if __name__ == "__main__":
    out_path = os.path.join(cfg["work"], "vad.json")
    tracks = [t["id"] for t in json.load(open(os.path.join(cfg["work"], "tracks.json")))]
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = [t for t in tracks if t not in out]
    if todo:
        print(f"VAD on {len(todo)} tracks with {os.cpu_count()} processes ...", flush=True)
    with ProcessPoolExecutor(min(len(todo), os.cpu_count()) or 1) as ex:
        for done, f in enumerate(as_completed([ex.submit(run, t) for t in todo]), 1):
            track_id, segs = f.result()
            out[track_id] = segs
            print(f"vad {done}/{len(todo)} {track_id}: {len(segs)} speech segments", flush=True)
    json.dump({t: out[t] for t in tracks}, open(out_path, "w"))
