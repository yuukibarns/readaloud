# /// script
# requires-python = "==3.12.*"
# dependencies = ["kaggle>=2.2"]
# ///
"""Step 2 on Kaggle, from the terminal: upload WORK/bundle.zip as a private dataset, run readaloud_align.ipynb on
Kaggle's GPU T4 x2, and download results.zip into WORK.

usage: uv run kaggle_run.py WORK_DIR [--download | --upload-only]
  --upload-only: upload the bundle and the notebook, then print the notebook's editor link - Run All there gets a GPU
              within seconds, while background runs (the default) can wait in Kaggle's queue for half an hour or more
  --download: don't upload or start anything, just wait for the latest run and fetch its results.zip
              (e.g. after the laptop slept and this script was interrupted; the run itself goes on at Kaggle)
Needs a Kaggle API token in ~/.kaggle/access_token (kaggle.com/settings/api → Generate New Token) and a
phone-verified Kaggle account (GPU and internet access). Dataset and notebook are private, named after WORK_DIR.
Re-running after a failure skips the upload if bundle.zip hasn't changed.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError

W = os.path.abspath(sys.argv[1])
DOWNLOAD_ONLY = "--download" in sys.argv[2:]
UPLOAD_ONLY = "--upload-only" in sys.argv[2:]
HERE = os.path.dirname(os.path.abspath(__file__))
bundle = f"{W}/bundle.zip"
if not os.path.exists(bundle):
    sys.exit(f"no {bundle} - run prepare.py first")

api = KaggleApi()
api.authenticate()
user = api.get_config_value("username")
slug = "readaloud-" + re.sub(r"[^a-z0-9]+", "-", os.path.basename(W).lower()).strip("-")
dataset, kernel = f"{user}/{slug}", f"{user}/{slug}-align"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if DOWNLOAD_ONLY:
    print(f"waiting for the latest run of https://www.kaggle.com/code/{kernel} ...", flush=True)
# --- 1. the bundle as a private dataset (skipped when this exact bundle is already up) ---------------------------
up = f"{W}/.kaggle-dataset"
os.makedirs(up, exist_ok=True)
digest, sent = sha256(bundle), f"{W}/.kaggle-dataset.sha256"
def status_of(ref):
    """The dataset's processing status, or None while Kaggle doesn't (yet) know it (it answers 403/404 then)."""
    try:
        return str(api.dataset_status(ref)).lower()
    except HTTPError:
        return None


exists = status_of(dataset) is not None
if DOWNLOAD_ONLY:
    pass                                            # nothing to upload: only fetch the latest run's results
elif exists and os.path.exists(sent) and open(sent).read() == digest:
    print(f"dataset {dataset}: this bundle is already uploaded")
else:
    staged = f"{up}/bundle.zip"
    if os.path.exists(staged):
        os.remove(staged)
    try:
        os.link(bundle, staged)                      # no copy of 100+ MB when on the same disk
    except OSError:
        shutil.copy(bundle, staged)
    json.dump({"title": slug, "id": dataset, "licenses": [{"name": "unknown"}]},
              open(f"{up}/dataset-metadata.json", "w"))
    print(f"uploading {os.path.getsize(bundle) / 1048576:.0f} MB to dataset {dataset} ...", flush=True)
    resp = None
    if not exists:
        resp = api.dataset_create_new(up, public=False, convert_to_csv=False)
        if resp.error and "already in use" in resp.error:   # created by an earlier attempt: add a version instead
            exists = True
    if exists:
        resp = api.dataset_create_version(up, "new bundle", convert_to_csv=False)    # old versions stay: a notebook is pinned to one
    if resp.error:
        sys.exit(f"Kaggle refused the dataset: {resp.error}")
    print(f"dataset: {getattr(resp, 'url', '') or 'https://www.kaggle.com/datasets/' + dataset}", flush=True)
    open(sent, "w").write(digest)
status, t_wait = None, time.time()
while not DOWNLOAD_ONLY:                            # Kaggle processes the upload before a notebook can use it
    now = status_of(dataset)
    if now != status:
        print(f"  dataset {now or 'not visible yet'}", flush=True)
        status = now
    if status and "ready" in status:
        break
    if time.time() - t_wait > 1800:
        sys.exit(f"dataset {dataset} is still '{status or 'not visible'}' after 30 min - check "
                 f"https://www.kaggle.com/datasets/{dataset} and run this again")
    time.sleep(15)

# --- 2. run the notebook on GPU T4 x2 ------------------------------------------------------------------------------
kd = f"{W}/.kaggle-kernel"
os.makedirs(kd, exist_ok=True)
shutil.copy(f"{HERE}/readaloud_align.ipynb", f"{kd}/readaloud_align.ipynb")
json.dump({"id": kernel, "title": f"{slug}-align", "code_file": "readaloud_align.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [dataset], "competition_sources": [],
           "kernel_sources": [], "model_sources": []}, open(f"{kd}/kernel-metadata.json", "w"))
if UPLOAD_ONLY:
    # save the notebook without starting a background run, if this version of the Kaggle library can
    try:
        api.kernels_push(kd, no_run=True)
    except TypeError:
        pass                                    # older library: the notebook is unchanged anyway, nothing to save
    print(f"uploaded. Open https://www.kaggle.com/code/{kernel}/edit , check Accelerator: GPU T4 x2 and that the "
          f"input {dataset} is its latest version, Run All, then download results.zip from the Output panel into {W}")
    sys.exit(0)
if not DOWNLOAD_ONLY:
    resp = api.kernels_push(kd)
    if getattr(resp, "error", None):
        sys.exit(f"Kaggle refused the notebook: {resp.error}")
    print(f"running on Kaggle: {getattr(resp, 'url', '') or 'https://www.kaggle.com/code/' + kernel}", flush=True)

t0, last = time.time(), None
while True:
    if last is not None or not DOWNLOAD_ONLY:
        time.sleep(30)
    try:
        r = api.kernels_status(kernel)
    except Exception as e:                      # a network hiccup (e.g. after sleep): keep waiting
        print(f"  (status check failed: {e}; retrying)", flush=True)
        last = last or "unknown"
        continue
    status = str(r.status).lower()
    if status != last:
        print(f"  {time.time() - t0:5.0f}s  {status}", flush=True)
        last = status
    if "complete" in status:
        break
    if "error" in status or "cancel" in status:
        log_dir = f"{W}/.kaggle-log"
        try:
            api.kernels_output(kernel, log_dir, file_pattern=r".*\.log$", force=True)
            for name in os.listdir(log_dir):
                print(open(f"{log_dir}/{name}", errors="replace").read()[-4000:])
        except Exception as e:
            print(f"(couldn't fetch the log: {e})")
        sys.exit(f"the Kaggle run failed: {getattr(r, 'failure_message', '') or status} - see "
                 f"https://www.kaggle.com/code/{kernel}")

# --- 3. results.zip into WORK ---------------------------------------------------------------------------------------
out = f"{W}/.kaggle-output"
shutil.rmtree(out, ignore_errors=True)
api.kernels_output(kernel, out, file_pattern=r"(^|/)results\.zip$", force=True)
found = [os.path.join(d, f) for d, _, fs in os.walk(out) for f in fs if f == "results.zip"]
if not found:
    sys.exit(f"the run finished but produced no results.zip - see https://www.kaggle.com/code/{kernel}")
if os.path.exists(f"{W}/results.zip"):
    os.replace(f"{W}/results.zip", f"{W}/results.prev.zip")
shutil.move(found[0], f"{W}/results.zip")
print(f"done in {(time.time() - t0) / 60:.1f} min: {W}/results.zip - now run: uv run build.py {W}")
