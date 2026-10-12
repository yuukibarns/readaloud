# /// script
# requires-python = "==3.12.*"
# dependencies = ["torch", "silero-vad", "soundfile", "fugashi", "unidic-lite"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Step 1 (local): chapter tracks, Deepgram word timings, VAD, chunk plan; packs bundle.zip for the GPU notebook.

usage: python prepare.py WORK_DIR      (run it with the Python that has torch + silero-vad + soundfile)
WORK_DIR/book.json: {"epub": "...", "audio": ["part1.m4b", "part2.m4b"], "language": "Japanese"}
Deepgram key: $DEEPGRAM_API_KEY or ~/.config/readaloud/deepgram_key
Every step is cached in WORK_DIR, so re-running only redoes what is missing.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from book import LANG_CODES, load_config  # noqa: E402

cfg = load_config(os.path.abspath(sys.argv[1]))
W = cfg["work"]
try:                                     # check before spending anything on Deepgram
    import silero_vad, soundfile, torch  # noqa: F401
except ImportError as e:
    sys.exit(f"{e.name} is missing: run prepare.py with the Python that has torch, silero-vad and soundfile installed")
os.makedirs(f"{W}/tracks", exist_ok=True)
os.makedirs(f"{W}/deepgram", exist_ok=True)

# --- tracks: one per chapter mark (or one per file if it has none) --------------------------------
tracks = []
for p, f in enumerate(cfg["audio"], 1):
    src = f if os.path.isabs(f) else os.path.join(W, f)
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json", src],
                                      capture_output=True, text=True, check=True).stdout)
    chapters = probe.get("chapters") or [{"start_time": 0, "end_time": probe["format"]["duration"], "tags": {"title": f}}]
    for i, c in enumerate(chapters, 1):
        tracks.append({"id": f"{p}-{i:02}", "src": src, "start": float(c["start_time"]), "end": float(c["end_time"]),
                       "title": c.get("tags", {}).get("title", "")})
json.dump(tracks, open(f"{W}/tracks.json", "w"), ensure_ascii=False, indent=1)


def to_ogg(t):
    out = f"{W}/tracks/{t['id']}.ogg"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t["start"]), "-to", str(t["end"]), "-i", t["src"],
                        "-map", "0:a", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "24k", out + ".tmp.ogg"], check=True)
        os.rename(out + ".tmp.ogg", out)


print(f"{len(tracks)} tracks, {sum(t['end'] - t['start'] for t in tracks) / 3600:.1f} h of audio - extracting ...", flush=True)
with ThreadPoolExecutor(6) as ex:
    for done, _ in enumerate(as_completed([ex.submit(to_ogg, t) for t in tracks]), 1):
        print(f"\r  tracks extracted: {done}/{len(tracks)}", end="", flush=True)
print()

# --- Deepgram rough transcript with word timings ------------------------------------------------------
key = os.environ.get("DEEPGRAM_API_KEY") or open(os.path.expanduser("~/.config/readaloud/deepgram_key")).read().strip()
url = f"https://api.deepgram.com/v1/listen?model=nova-3&language={LANG_CODES[cfg['language']]}&punctuate=true"


def transcribe(t):
    out = f"{W}/deepgram/{t['id']}.json"
    if os.path.exists(out):
        return t["id"], "cached"
    data = open(f"{W}/tracks/{t['id']}.ogg", "rb").read()
    err = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, data=data, headers={"Authorization": f"Token {key}", "Content-Type": "audio/ogg"})
            body = urllib.request.urlopen(req, timeout=1800).read()
            json.loads(body)
            open(out, "wb").write(body)
            return t["id"], "ok"
        except Exception as e:
            err = e
            time.sleep(10)
    return t["id"], f"FAILED: {err}"


with ThreadPoolExecutor(3) as ex:
    for done, f in enumerate(as_completed([ex.submit(transcribe, t) for t in tracks]), 1):
        tid, status = f.result()
        print(f"deepgram {done}/{len(tracks)} {tid} {status}", flush=True)

failed = [t["id"] for t in tracks if not os.path.exists(f"{W}/deepgram/{t['id']}.json")]
if failed:
    sys.exit(f"Deepgram failed for {failed} - run prepare.py again to retry")

# --- VAD and the chunk plan (local CPU; the GPU machine only aligns) ------------------------------------
for script in ("vad.py", "plan.py"):
    subprocess.run([sys.executable, f"{HERE}/{script}", W], check=True)

# --- bundle for the notebook --------------------------------------------------------------------------
bundle_cfg = {k: v for k, v in cfg.items() if k not in ("work", "epub_path", "audio")}
bundle_cfg["epub"] = "book.epub"
with zipfile.ZipFile(f"{W}/bundle.zip", "w") as z:
    z.writestr("book.json", json.dumps(bundle_cfg, ensure_ascii=False))
    z.write(cfg["epub_path"], "book.epub")
    z.writestr("tracks.json", json.dumps([{k: v for k, v in t.items() if k != "src"} for t in tracks], ensure_ascii=False))
    z.write(f"{W}/plan.json", "plan.json")
    for t in tracks:
        z.write(f"{W}/tracks/{t['id']}.ogg", f"tracks/{t['id']}.ogg")
    for script in ("book.py", "align.py"):
        z.write(f"{HERE}/{script}", f"scripts/{script}")
print(f"wrote {W}/bundle.zip ({os.path.getsize(f'{W}/bundle.zip') / 1048576:.0f} MB) - upload it to Kaggle or Colab")
