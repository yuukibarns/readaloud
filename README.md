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
| 2. Precise alignment | **Colab / Kaggle GPU** (or your PC's CPU) | `readaloud_align.ipynb` (runs `align.py`) | T4 GPU: ~8 min; CPU: ~1 h 40 min |
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

3. Open `readaloud_align.ipynb` on Colab or Kaggle (instructions are in its first cell), give it `bundle.zip`,
   run all cells, and download `results.zip`.

   No GPU at hand? Run it locally instead (CPU, about 9× real time):
   `uv run align.py WORK`

4. Put `results.zip` into `WORK` (no need to unzip it), then `uv run build.py WORK` → `WORK/<title> (readaloud).epub`.

Python packages: each script declares its own dependencies inline (PEP 723), so `uv run` creates its environment on
first use (Python 3.12, CPU-only PyTorch). Scripts can be run from any directory: `uv run /path/to/readaloud/prepare.py WORK`.
On Colab/Kaggle the notebook installs what `align.py` needs into the platform's Python instead.

## What each script does

- `book.py` — reads the EPUB's spine, splits `<p>` paragraphs into sentences (at 。！？, keeping 「…」と言った together;
  never inside markup; furigana `<rt>` is shown but not treated as spoken text).
- `prepare.py` — tracks from chapter marks (16 kHz mono Opus), Deepgram nova-3 word timings, then `vad.py` and `plan.py`, then the bundle.
- `vad.py` — Silero VAD speech segments per track; the pauses are the only places audio gets cut.
- `plan.py` — locates every sentence in Deepgram's transcript by *reading*: both sides go through a Japanese
  morphological analyser (`reading.py`: fugashi + UniDic, digits as kanji numerals), so 「１２」/十二 and 鼠/ねずみ
  compare equal. Unique 10-character readings chained in order are anchors; between anchors an edit-distance
  alignment with affine gaps marks each book character heard, misheard or not read (skipped text is one long
  deletion). A sentence is heard if its dominant run (heard characters lying together in the transcript, as in
  Storyteller's aligner) covers at least half of its reading; short lines Deepgram missed between heard neighbours
  are kept if the time there could hold them at twice the narrator's measured pace. Tracks are cut into ≤160 s
  chunks, only before heard sentences, in the first silence after the previous sentence's last heard word.
  Finally every chunk is checked both ways: how much of its text was heard inside its audio, and how much of its
  audio is its text. The chunks that agree least are printed, so a text/audio mismatch shows before the GPU run.
- `align.py` — Qwen3-ForcedAligner on each chunk with its exact sentences (batched on GPU, resumable). On a GPU it runs
  in float16: on a whole 15 h book it gave the same sentence times as float32 (a few moved by one 80 ms step), 3.5× faster.
  Also writes `units.json`, the aligner's raw word units per chunk, which the notebook packs into `results.zip`.
- `compare.py` — compares two alignment runs, e.g. `--dtype float32` vs float16.
- `book.py` — the text that is aligned: every `<p>` and `<h1>`–`<h6>` in the spine, except the table-of-contents
  pages the book declares (EPUB 3 nav, EPUB 2 guide `type="toc"`).
- `build.py` — refuses timings aligned from a different `plan.json` (sentence ids are book positions); sentence `<span>`s, one SMIL per document, 64 kbps mono AAC per track, EPUB 3 OPF and nav.
  Each clip starts 0.15 s before the first sound. Where the aligner squeezes quick short lines (でも！, 「うん。) into an
  instant, the span is shared with the next sentence by text length; an end that overruns the next start is cut to it.

## Known limits

- Text that the narrator reads but the EPUB doesn't contain (or the reverse) makes the 1–3 neighbouring sentences'
  timing stretch over the difference. Use an EPUB of the same edition as the audiobook.
- Paragraphs must be `<p>` elements (true for almost all commercial and calibre-converted EPUBs).
- Sentence splitting rules are written for Japanese.
