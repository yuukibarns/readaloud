# Readaloud toolkit

Turns an EPUB plus its audiobook into a **readaloud EPUB 3** (Media Overlays): every sentence is highlighted while the
narrator reads it, and tapping a sentence plays it from its first syllable. Tested in Storyteller Reader (Android);
Media Overlays are also supported by Thorium (desktop) and Apple Books. Readest on Android played its own TTS voice
instead of the recording in our test.

Sentence timing comes from a forced aligner (Qwen3-ForcedAligner) that is given the book's exact text, so starts are
typically within a few hundredths of a second of the spoken sentence. Built and tested for Japanese.

## Pipeline

| Step | Where | Script | Time for a 15 h book |
|---|---|---|---|
| 1. Chapter tracks, Deepgram rough transcript, VAD, chunk plan; pack `bundle.zip` | your PC | `prepare.py` (runs `vad.py`, `plan.py`) | ~3 min + Deepgram upload (~140 MB; ~10 min through a slow proxy) |
| 2. Precise alignment | **Kaggle / Colab GPU** (or your PC's CPU) | `kaggle_upload.py` + the notebook, or `readaloud_align.ipynb` by hand (runs `align.py`) | Kaggle T4 x2: ~4 min + upload; one T4: ~8 min; CPU: ~1 h 40 min |
| 3. Build the EPUB | your PC | `build.py` | ~3 min |

Only step 2 needs a GPU, so the notebook does nothing else. Step 1 also prints how well the book matched the audio,
so a bad EPUB shows up before anything is uploaded.

## How to use

1. Make a work folder with a `book.json`:

   ```json
   {"epub": "/path/to/book.epub",
    "audio": ["/path/to/part1.m4b", "/path/to/part2.m4b"],
    "language": "Japanese"}
   ```

   `audio` lists the files in reading order. Chapter marks inside the files become tracks; files without marks are
   one track each. Optional: `"title"` (overrides the EPUB's title), `"spine"` (list of document paths to align).

2. `uv run prepare.py WORK` — needs ffmpeg and a Deepgram key (`$DEEPGRAM_API_KEY`, or the file
   `~/.config/readaloud/deepgram_key`). Writes `WORK/bundle.zip`. Every step is cached, so re-running is cheap.

3. **Kaggle (two GPUs):** `uv run kaggle_upload.py WORK` only uploads: `bundle.zip` goes up once, as its own
   private dataset (one file, `bundle.bin`, named `readaloud-<book>-<sha8>` after its fingerprint). Upload books
   whenever they're ready, also while the notebook aligns another one. All books use one notebook,
   `readaloud-align`, which the first upload creates without running it. Open the printed link; at the top of step 2
   set `BOOK` to the book's folder name (empty: the bundle uploaded last; a full dataset name picks an exact bundle),
   check *Accelerator: GPU T4 x2*, press *Run All*: step 2 lists every uploaded bundle, downloads the chosen one,
   checks its fingerprint and aligns it. Download `results.zip` from the *Output* panel. Keep the notebook open for
   the next book: the GPU session stays yours (a new session can wait in Kaggle's queue). `--clean` also deletes
   the book's datasets for other bundles; `--notebook` saves the repo's notebook as a new version of
   `readaloud-align` after it changed (reload an open editor to get it). One-time setup: a phone-verified Kaggle
   account and an API token from kaggle.com/settings/api (*Generate New Token*) saved as `~/.kaggle/access_token`
   (the notebook uses its own Kaggle login). Free accounts get 30 GPU hours a week.

   **Or by hand on Colab (one T4):** open `readaloud_align.ipynb` there (instructions are in its first cell:
   https://colab.research.google.com/github/yuukibarns/readaloud/blob/main/readaloud_align.ipynb), give it
   `bundle.zip`, run all cells, and download `results.zip`.

   No GPU at hand? Run it locally instead (CPU, about 9× real time):
   `uv run align.py WORK`

4. Put `results.zip` into `WORK` if you downloaded it by hand (no need to unzip it), then `uv run build.py WORK` → `WORK/<title> (readaloud).epub`.

Python packages: each script declares its own dependencies inline (PEP 723), so `uv run` creates its environment on
first use (Python 3.12, CPU-only PyTorch). Scripts can be run from any directory: `uv run /path/to/readaloud/prepare.py WORK`.
On Colab/Kaggle the notebook installs what `align.py` needs into the platform's Python instead.

## What each script does

- `book.py` — reads the EPUB's spine, splits `<p>` paragraphs into sentences (at 。！？, keeping 「…」と言った together;
  never inside markup; furigana `<rt>` is shown but not treated as spoken text).
- `prepare.py` — tracks from chapter marks (16 kHz mono Opus), Deepgram nova-3 word timings, then `vad.py` and `plan.py`, then the bundle.
- `vad.py` — Silero VAD speech segments per track, edges to the millisecond (Silero's default rounds them to
  0.1 s). Its per-frame probabilities are kept in `WORK/vad/`, so changing the settings never runs the model again;
  at most 4 processes, each reading its track in blocks.
- `plan.py` — locates every sentence in Deepgram's transcript by *reading*: both sides go through a Japanese
  morphological analyser (`reading.py`: fugashi + UniDic, digits as kanji numerals), so 「１２」/十二 and 鼠/ねずみ
  compare equal. Unique 10-character readings chained in order are anchors; between anchors an edit-distance
  alignment with affine gaps marks each book character heard, misheard or not read (skipped text is one long
  deletion). A sentence is heard if its dominant run (heard characters lying together in the transcript, as in
  Storyteller's aligner) covers at least half of its reading; short lines Deepgram missed between heard neighbours
  are kept if the time there could hold them at twice the narrator's measured pace. Tracks are cut into ≤160 s
  chunks, only before heard sentences, in the first silence after the previous sentence's last heard word.
  The pauses are defined once, here, for the whole pipeline: VAD's pauses minus any stretch where Deepgram heard a
  word starting inside them (VAD misses short and soft speech: a lone 「へ？」, a whispered line). `plan.json`
  carries the resulting speech map; chunks are cut, pauses shortened (`align.py`) and sentence boundaries placed
  (`build.py`) only in these pauses.
  Finally every chunk is checked both ways: how much of its text was heard inside its audio, and how much of its
  audio is its text. The chunks that agree least are printed, so a text/audio mismatch shows before the GPU run.
  Speech in a chunk that belongs to no text (a picture read aloud, e.g. a newspaper advert or a title set as an
  image; an announcement) is recorded as `extra_speech`: speech whose every transcribed character the alignment
  found no book text for, with no sentence located over it. `align.py` leaves it out of the aligner's audio.
- `align.py` — Qwen3-ForcedAligner on each chunk with its exact sentences, as spoken (`book.spoken_text`: NFKC,
  digits as kanji numerals, so 「７」 is aligned as 七). The aligner doesn't model silence, so it never gets long
  pauses: every pause in `plan.json`'s speech map is shortened to 0.2 s before aligning, and word times are mapped back (a start in a
  shortened pause goes to where speech resumes, an end to where it stopped). (batched on GPU, resumable; with several
  GPUs one worker process each, every track on one GPU). On a GPU it runs
  in float16: on a whole 15 h book it gave the same sentence times as float32 (a few moved by one 80 ms step), 3.5× faster.
  Also writes `units.json`, the aligner's raw word units per chunk, which the notebook packs into `results.zip`.
- `kaggle_upload.py` — step 2 on Kaggle, the upload: the bundle (skipped if already there) to its own dataset; creates
  the notebook `readaloud-align` once, without running it (Kaggle's `--no-run`, from the Kaggle library pinned to
  the commit that added it); prints the editor link and the `BOOK` to set.
- `compare.py` — compares two alignment runs, e.g. `--dtype float32` vs float16.
- `book.py` — the text that is aligned: every `<p>` and `<h1>`–`<h6>` in the spine, except the table-of-contents
  pages the book declares (EPUB 3 nav, EPUB 2 guide `type="toc"`).
- `build.py` — prints a quality report (also `WORK/report.txt`): coverage, silent headings, clip sanity, silence
  before a tapped sentence, narration outside every clip, agreement with Deepgram and the largest disagreements.
  Refuses results aligned from a different `plan.json` (sentence ids are book positions) or by a
  different `align.py` (results carry its fingerprint); sentence `<span>`s, one SMIL per document, 64 kbps mono AAC per track, EPUB 3 OPF and nav.
  Sentence boundaries combine both sources: the aligner doesn't model silence (after a pause a word often "starts"
  where the previous one ended), but it is sure a sentence is still being spoken when its last word starts and the
  next has begun when its second word starts; the first pause in between is the boundary (one sentence ends
  where it begins, the next starts where speech resumes). The aligner also collapses a short word spoken alone
  before a pause (a section number 「七」, 「部活、…」) onto the words after it; if Deepgram heard a sentence's first
  sound inside a speech segment that no sentence covers, the sentence starts there. Each clip starts 0.15 s
  before its start. Where the aligner squeezes quick short lines (でも！, 「うん。) into an
  instant, the span is shared with the next sentence by text length; an end that overruns the next start is cut to it.

## Known limits

- Text that the narrator reads but the EPUB doesn't contain (or the reverse) makes the 1–3 neighbouring sentences'
  timing stretch over the difference. Use an EPUB of the same edition as the audiobook.
- Paragraphs must be `<p>` elements (true for almost all commercial and calibre-converted EPUBs).
- Sentence splitting rules are written for Japanese.
