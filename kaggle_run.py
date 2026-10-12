# /// script
# requires-python = "==3.12.*"
# dependencies = ["kaggle @ git+https://github.com/Kaggle/kaggle-cli@9c57dd74c29904c40ff7879eed33644be89d2e2f"]
# ///
"""Step 2 on Kaggle: upload WORK/bundle.zip and set up an alignment notebook for it; you press Run All.

usage: uv run kaggle_run.py WORK_DIR [--clean]
Then open the printed link, check Accelerator: GPU T4 x2, Run All (an interactive session gets a GPU within seconds;
background runs could wait in Kaggle's queue for an hour), and download results.zip from the Output panel into
WORK_DIR.
  --clean: also delete this book's datasets and notebooks for other bundles

Needs a Kaggle API token in ~/.kaggle/access_token (kaggle.com/settings/api → Generate New Token) and a
phone-verified Kaggle account (GPU and internet access).

Each bundle gets its own private dataset and its own notebook, both named after the book and the bundle's
fingerprint (readaloud-<book>-<sha8>, readaloud-<book>-<sha8>-align). The bundle goes up as one file, bundle.bin
(Kaggle unpacks a .zip into many files, which then take long to attach). The notebook is saved without running
(Kaggle's Quick Save), with that dataset as input and the fingerprint written in. Being new, it has no editor draft
that could hold older code or another input. Re-running with the same bundle reuses both.

(The Kaggle library is pinned to the commit that added --no-run, not yet in a release; Kaggle calls are retried
by the library when the connection drops.)
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
CLEAN = "--clean" in sys.argv[2:]
HERE = os.path.dirname(os.path.abspath(__file__))
bundle = f"{W}/bundle.zip"
if not os.path.exists(bundle):
    sys.exit(f"no {bundle} - run prepare.py first")

api = KaggleApi()
api.authenticate()
call = api.with_retry                                   # retries when the connection drops
user = api.get_config_value("username")
book = "readaloud-" + re.sub(r"[^a-z0-9]+", "-", os.path.basename(W).lower()).strip("-")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def status_of(ref):
    """A dataset's processing status, or None while Kaggle doesn't (yet) know it (it answers 403/404 then)."""
    try:
        return str(call(api.dataset_status)(ref)).lower()
    except HTTPError:
        return None


fingerprint = sha256(bundle)[:8]
slug = f"{book}-{fingerprint}"
dataset, kernel = f"{user}/{slug}", f"{user}/{slug}-align"

# --- 1. this bundle as its own dataset, one file ----------------------------------------------------------------
if status_of(dataset) is not None:
    print(f"dataset {dataset}: this bundle is already uploaded")
else:
    up = f"{W}/.kaggle-dataset"
    shutil.rmtree(up, ignore_errors=True)
    os.makedirs(up)
    try:
        os.link(bundle, f"{up}/bundle.bin")              # no copy of 100+ MB when on the same disk
    except OSError:
        shutil.copy(bundle, f"{up}/bundle.bin")
    json.dump({"title": slug, "id": dataset, "licenses": [{"name": "unknown"}]},
              open(f"{up}/dataset-metadata.json", "w"))
    print(f"uploading {os.path.getsize(bundle) / 1048576:.0f} MB to dataset {dataset} ...", flush=True)
    resp = api.dataset_create_new(up, public=False, convert_to_csv=False)
    if resp.error:
        sys.exit(f"Kaggle refused the dataset: {resp.error}")
    shutil.rmtree(up, ignore_errors=True)
status, t_wait = None, time.time()
while True:                                             # Kaggle processes the upload before a notebook can use it
    now = status_of(dataset)
    if now != status:
        print(f"  dataset {now or 'not visible yet'}", flush=True)
        status = now
    if status and "ready" in status:
        break
    if time.time() - t_wait > 1800:
        sys.exit(f"dataset {dataset} is still '{status or 'not visible'}' after 30 min - check "
                 f"https://www.kaggle.com/datasets/{dataset} and run this again")
    time.sleep(10)

# --- 2. this bundle's notebook, saved without running ----------------------------------------------------------
kd = f"{W}/.kaggle-kernel"
shutil.rmtree(kd, ignore_errors=True)
os.makedirs(kd)
nb = json.load(open(f"{HERE}/readaloud_align.ipynb"))
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if 'EXPECTED_BUNDLE = ""' in src:
        cell["source"] = src.replace('EXPECTED_BUNDLE = ""', f'EXPECTED_BUNDLE = "{fingerprint}"').splitlines(True)
json.dump(nb, open(f"{kd}/readaloud_align.ipynb", "w"), ensure_ascii=False, indent=1)
json.dump({"id": kernel, "title": f"{slug}-align", "code_file": "readaloud_align.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
           "machine_shape": "NvidiaTeslaT4", "dataset_sources": [dataset], "competition_sources": [],
           "kernel_sources": [], "model_sources": []}, open(f"{kd}/kernel-metadata.json", "w"))
resp = call(api.kernels_push)(kd, no_run=True)
if getattr(resp, "error", None):
    sys.exit(f"Kaggle refused the notebook: {resp.error}")

# --- 3. on request: this book's datasets and notebooks for other bundles ----------------------------------------
if CLEAN:
    own = re.compile(re.escape(book) + r"(-[0-9a-f]{8})?")
    for d in call(api.dataset_list)(mine=True, search=book):
        name = d.ref.split("/")[-1]
        if name != slug and own.fullmatch(name):
            call(api.dataset_delete)(user, name, no_confirm=True)
            print(f"  deleted the dataset {user}/{name}")
    for k in call(api.kernels_list)(mine=True, search=book):
        name = k.ref.split("/")[-1]
        if name != f"{slug}-align" and name.endswith("-align") and own.fullmatch(name[:-len("-align")]):
            call(api.kernels_delete)(k.ref, no_confirm=True)
            print(f"  deleted the notebook {k.ref}")

print(f"""
ready. Open https://www.kaggle.com/code/{kernel}/edit
  - Settings: Accelerator GPU T4 x2 (the input {slug} is attached)
  - Run All
  - when it's done, download results.zip from the Output panel into {W}
then: uv run build.py {W}""")
