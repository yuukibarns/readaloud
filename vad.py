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

usage: python vad.py WORK_DIR      (CPU work, one process per core as free memory allows; ~300 MB each)
Silero's speech probability for every 32 ms frame is kept in WORK/vad/<track>.npy; vad.json (the segments) is made
from those with SETTINGS, so changing the settings never runs the model again.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book import load_config, track_wav  # noqa: E402

cfg = load_config(os.path.abspath(sys.argv[1]))
PROBS = os.path.join(cfg["work"], "vad")


def available_memory():
    """Bytes of memory free for new work (Linux /proc/meminfo; elsewhere assume 4 GB)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 4 << 30


# a worker needs ~300 MB (275 MB measured on a 106 min track): use at most half the free memory at 500 MB each,
# and leave two cores for the rest of the computer
WORKERS = max(1, min(os.cpu_count() - 2, available_memory() // 2 // (500 << 20)))
# edges to the millisecond (silero rounds to 0.1 s by default, which clips word edges and drops short segments)
SETTINGS = dict(min_silence_duration_ms=120, speech_pad_ms=0, time_resolution=3)


def probabilities(track_id):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # torch.jit.load deprecation inside silero-vad
    import numpy as np
    import soundfile
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    model = load_silero_vad()
    model.reset_states()
    probs, n = [], 0
    # read in blocks (a whole track is ~1 GB as float32; one process per core would exhaust memory)
    # inference_mode, as silero's get_speech_timestamps uses no_grad: else autograd keeps every step of the model's
    # recurrent state, and memory grows with the track (a 106 min track passed 1.7 GB)
    with torch.inference_mode():
        for block in soundfile.blocks(track_wav(cfg, track_id), blocksize=512 * 4096, dtype="float32"):
            n += len(block)
            block = torch.from_numpy(block)
            for i in range(0, len(block), 512):                 # what silero's get_speech_timestamps does
                chunk = block[i:i + 512]
                if len(chunk) < 512:                            # only at the very end (blocks are 512-aligned)
                    chunk = torch.nn.functional.pad(chunk, (0, 512 - len(chunk)))
                probs.append(model(chunk, 16000).item())
    tmp = f"{PROBS}/{track_id}.tmp.npy"                         # written whole, then renamed: a run killed while
    np.save(tmp, np.array(probs, dtype=np.float32))             # saving never leaves a half file that counts as done
    os.replace(tmp, f"{PROBS}/{track_id}.npy")
    return track_id, n


def segments(track_id, n_samples):
    import numpy as np
    from silero_vad.utils_vad import get_speech_timestamps_from_probs
    probs = np.load(f"{PROBS}/{track_id}.npy").tolist()
    segs = get_speech_timestamps_from_probs(probs, sampling_rate=16000, return_seconds=True,
                                            audio_length_samples=n_samples, **SETTINGS)
    return [[s["start"], s["end"]] for s in segs]


if __name__ == "__main__":
    import soundfile
    os.makedirs(PROBS, exist_ok=True)
    tracks = [t["id"] for t in json.load(open(os.path.join(cfg["work"], "tracks.json")))]

    def complete(t):
        """This track's probabilities are saved, one per 512-sample frame of its audio."""
        import numpy as np
        try:
            return len(np.load(f"{PROBS}/{t}.npy", mmap_mode="r")) == -(-soundfile.info(track_wav(cfg, t)).frames // 512)
        except (OSError, ValueError):
            return False

    todo = [t for t in tracks if not complete(t)]
    if todo:
        workers = min(len(todo), WORKERS)
        print(f"VAD on {len(todo)} tracks with {workers} process{'es' if workers > 1 else ''} ...", flush=True)
        with ProcessPoolExecutor(workers) as ex:
            for done, f in enumerate(as_completed([ex.submit(probabilities, t) for t in todo]), 1):
                print(f"vad {done}/{len(todo)} {f.result()[0]}", flush=True)
    out = {t: segments(t, soundfile.info(track_wav(cfg, t)).frames) for t in tracks}
    json.dump(out, open(os.path.join(cfg["work"], "vad.json"), "w"))
    print(f"vad.json: {sum(map(len, out.values()))} speech segments", flush=True)
