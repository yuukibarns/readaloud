# /// script
# requires-python = "==3.12.*"
# dependencies = []
# ///
"""Step 3 (local): build the readaloud EPUB 3 from timings.json.

usage: python build.py WORK_DIR [OUTPUT.epub]      (reads timings from WORK_DIR/results.zip or timings.json)
Adds sentence <span>s, one SMIL media overlay per text document, one AAC file per track, an EPUB 3 OPF and nav.
"""
import bisect
import hashlib
import html
import json
import os
import posixpath
import re
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book import LANG_CODES, OPF_NS, Epub, book_sentences, load_config, paragraphs, sentence_times, text_docs  # noqa: E402

cfg = load_config(os.path.abspath(sys.argv[1]))
W = cfg["work"]
epub = Epub(cfg["epub_path"])
title = cfg.get("title") or (epub.opf.findtext(".//dc:title", namespaces=OPF_NS) or "book")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(W, f"{title} (readaloud).epub")
LANG = LANG_CODES[cfg["language"]]
LEAD = 0.15                     # start each clip just before the first sound, so replays never clip a syllable
tracks = {t["id"]: t for t in json.load(open(f"{W}/tracks.json"))}
track_order = list(tracks)
# timings from the GPU notebook: results.zip as downloaded, or an already extracted timings.json (whichever is newer)
results, extracted = f"{W}/results.zip", f"{W}/timings.json"
aligned_chunks, units = None, None
if os.path.exists(results) and (not os.path.exists(extracted) or os.path.getmtime(results) > os.path.getmtime(extracted)):
    z = zipfile.ZipFile(results)
    timings = json.loads(z.read("timings.json"))
    if "units.json" in z.namelist():
        units = json.loads(z.read("units.json"))
        aligned_chunks = [{k: c[k] for k in ("track", "start", "end", "sentences")} for c in units]  # see chunk_keys
    print("timings from results.zip")
else:
    timings = json.load(open(extracted))
sentences = book_sentences(cfg)
# results from a different aligner than this one (an older bundle, a stale notebook input) are refused
here_aligner = hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "align.py"),
                                   "rb").read()).hexdigest()[:12]
made_by = {c.get("aligner") for c in units} if units else {None}
if made_by != {here_aligner}:
    sys.exit(f"these results were made by a different align.py ({', '.join(sorted(str(m) for m in made_by))}; "
             f"this one is {here_aligner}) - run prepare.py, align the new bundle.zip, and use its results.zip")
if units:                               # the aligner's raw word units: map them onto sentences with today's book.py
    by_id = {s["id"]: s for s in sentences}
    timings = {t["id"]: {**t, "track": c["track"]} for c in units
               for t in sentence_times(c, c["units"], by_id)}
# sentence ids are positions in the book: results aligned from another plan would land on the wrong sentences
plan_chunks = json.load(open(f"{W}/plan.json"))["chunks"]
chunk_keys = ("track", "start", "end", "sentences")       # what defines a chunk (plan.json also records extra_speech)
if (aligned_chunks if aligned_chunks is not None else set(timings)) != \
        ([{k: c[k] for k in chunk_keys} for c in plan_chunks] if aligned_chunks is not None
         else {i for c in plan_chunks for i in c["sentences"]}):
    sys.exit("these timings were aligned from a different plan.json than the current one - "
             "upload the current bundle.zip, align again, and use its results.zip")

# --- clean per-sentence times: monotonic within a track, short gaps interpolated --------------------
times = []
for s in sentences:
    t = timings.get(s["id"])
    ok = t and t["start"] is not None and t["end"] >= t["start"]
    times.append({"track": t["track"], "start": t["start"], "end": t["end"], "last_start": t.get("last_start", t["end"]),
                  "second_start": t.get("second_start", t["start"])} if ok else None)
for k in range(1, len(times)):          # drop timings that go backwards
    p = next((times[j] for j in range(k - 1, -1, -1) if times[j]), None)
    if times[k] and p and p["track"] == times[k]["track"] and times[k]["start"] < p["start"]:
        times[k] = None
# Where one sentence ends and the next begins. The aligner doesn't model silence (after a pause a word often "starts"
# where the previous one ended), but it is sure the previous sentence is still being spoken when its last word starts,
# and the next one has begun when its second word starts. The pause between them, if any, is the first pause (in
# plan.py's speech map: VAD and Deepgram) after the former that begins before the latter: the previous sentence ends
# where it begins, the next starts where speech resumes. A track's first sentence starting inside a silence starts where that silence ends.
plan_speech = json.load(open(f"{W}/plan.json")).get("speech")
if plan_speech is None:
    sys.exit("plan.json is from an older prepare.py (no speech map) - run prepare.py again, then align again")
# speech that belongs to no text (plan.py's extra_speech) isn't the narration of any sentence: not speech here
extra = {(c["track"], tuple(x)) for c in plan_chunks for x in c.get("extra_speech", [])}
speech = {tr: [g for g in segs if (tr, (round(g[0], 3), round(g[1], 3))) not in extra] for tr, segs in plan_speech.items()}
SILENCE = {tr: [(segs[i][1], segs[i + 1][0]) for i in range(len(segs) - 1)] for tr, segs in speech.items()}
SPEECH_FROM = {tr: segs[0][0] for tr, segs in speech.items() if segs}
moved, prev = 0, None
for t in times:
    if not t:
        continue
    gaps = SILENCE.get(t["track"], [])
    if prev and prev["track"] == t["track"]:
        i = bisect.bisect_left([a for a, _ in gaps], prev["last_start"])
        if i < len(gaps) and gaps[i][0] <= t["second_start"] and gaps[i][1] < t["end"]:
            prev["end"] = min(prev["end"], gaps[i][0])
            moved += abs(t["start"] - gaps[i][1]) > 0.1     # (align.py's padding alone moves starts by 0.05 s)
            t["start"] = gaps[i][1]
    else:
        inside = [g for g in [(0.0, SPEECH_FROM.get(t["track"], 0.0))] + gaps if g[0] <= t["start"] < g[1]]
        if inside and inside[0][1] < t["end"]:
            moved += abs(t["start"] - inside[0][1]) > 0.1
            t["start"] = inside[0][1]
    prev = t
print(f"sentence boundaries placed in the pause between sentences: {moved} starts moved by more than 0.1 s")
nxt = None                              # an end that overruns the next sentence's start: trust the start
for k in range(len(times) - 1, -1, -1):
    if times[k]:
        if nxt and nxt["track"] == times[k]["track"]:
            times[k]["end"] = min(times[k]["end"], nxt["start"])
        nxt = times[k]
k = 0
while k < len(times):                   # the aligner squeezes quick short lines into an instant ("でも！", 「うん。」):
    j = k                               # share a run of those plus the sentence that got their time by text length
    while j + 1 < len(times) and times[j] and times[j + 1] and times[j + 1]["track"] == times[j]["track"] \
            and times[j]["end"] - times[j]["start"] < 0.1:
        j += 1
    if j > k:
        t0, span = times[k]["start"], times[j]["end"] - times[k]["start"]
        weights = [max(1, len(sentences[i]["norm"])) for i in range(k, j + 1)]
        for i, w in zip(range(k, j + 1), weights):
            times[i] = {"track": times[k]["track"], "start": t0, "end": t0 + span * w / sum(weights)}
            t0 = times[i]["end"]
    k = j + 1
for k, t in enumerate(times):           # still no duration (e.g. squeezed at the very end of a chunk): untimed
    if t and t["end"] - t["start"] < 0.05:
        times[k] = None
k = 0
while k < len(times):                   # fill runs of untimed sentences between two timed ones on a track
    if times[k] is None:
        j = k
        while j < len(times) and times[j] is None:
            j += 1
        prev, nxt = (times[k - 1] if k else None), (times[j] if j < len(times) else None)
        # (only if there is audio between the neighbours to give them)
        if prev and nxt and prev["track"] == nxt["track"] and 0.05 * (j - k) <= nxt["start"] - prev["end"] < 60:
            weights = [max(1, len(sentences[i]["norm"])) for i in range(k, j)]
            t0, span = prev["end"], nxt["start"] - prev["end"]
            for i, w in zip(range(k, j), weights):
                times[i] = {"track": prev["track"], "start": t0, "end": t0 + span * w / sum(weights)}
                t0 = times[i]["end"]
        k = j
    else:
        k += 1

clips = [None] * len(times)
for k, t in enumerate(times):
    if t:
        begin = t["start"] - LEAD
        if k and times[k - 1] and times[k - 1]["track"] == t["track"]:
            begin = max(begin, times[k - 1]["end"])
        clips[k] = {"track": t["track"], "begin": max(0.0, begin), "end": t["end"]}
for k, c in enumerate(clips):           # run each clip on to the next one when the pause between is short
    if c:
        nxt = clips[k + 1] if k + 1 < len(clips) else None
        same = nxt and nxt["track"] == c["track"]
        c["end"] = nxt["begin"] if same and nxt["begin"] - c["end"] < 2.0 else min(c["end"] + 0.4, nxt["begin"] if same else c["end"] + 0.4)
print(f"{sum(1 for c in clips if c)}/{len(clips)} sentences have audio")

# --- audio: one AAC file per track --------------------------------------------------------------------
os.makedirs(f"{W}/epub_audio", exist_ok=True)


def encode(tid):
    out = f"{W}/epub_audio/{tid}.m4a"
    if not os.path.exists(out):
        t = tracks[tid]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t["start"]), "-to", str(t["end"]), "-i", t["src"],
                        "-map", "0:a", "-ac", "1", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", out + ".tmp.m4a"], check=True)
        os.rename(out + ".tmp.m4a", out)


todo = [tid for tid in track_order if not os.path.exists(f"{W}/epub_audio/{tid}.m4a")]
if todo:
    print(f"encoding audio for {len(todo)} tracks ...", flush=True)
with ThreadPoolExecutor(6) as ex:
    for done, f in enumerate(as_completed([ex.submit(encode, tid) for tid in todo]), 1):
        f.result()
        print(f"\r  tracks encoded: {done}/{len(todo)}", end="", flush=True)
if todo:
    print()


def rel(from_file, to_path):
    return posixpath.relpath(to_path, posixpath.dirname(from_file))


# --- text documents with sentence spans, and one SMIL per document ------------------------------------
audio_dir = posixpath.join(epub.opf_dir, "audio")
smil_dir = posixpath.join(epub.opf_dir, "smil")
new_files, smil_for, smil_dur = {}, {}, {}
idx = 0
for f in text_docs(cfg, epub):
    doc = epub.read(f)
    smil = posixpath.join(smil_dir, posixpath.basename(f).rsplit(".", 1)[0] + ".smil")
    parts, last, pars = [], 0, []
    for m, pieces in paragraphs(doc):
        spans = []
        for piece, _ in pieces:
            sid = sentences[idx]["id"]
            spans.append(f'<span id="{sid}">{piece}</span>')
            c = clips[idx]
            if c:
                pars.append(f'<par id="p{sid}"><text src="{rel(smil, f)}#{sid}"/><audio src="{rel(smil, audio_dir + "/" + c["track"] + ".m4a")}" '
                            f'clipBegin="{c["begin"]:.3f}s" clipEnd="{c["end"]:.3f}s"/></par>')
            idx += 1
        parts.append(doc[last:m.start("inner")] + "".join(spans))
        last = m.end("inner")
    parts.append(doc[last:])
    doc = "".join(parts)
    # set the document language on the <html> start tag
    new_files[f] = re.sub(r"<html\b[^>]*>", lambda m: re.sub(r'\s(?:xml:)?lang="[^"]*"', "", m.group(0))
                          .replace("<html", f'<html lang="{LANG}" xml:lang="{LANG}"', 1), doc, count=1)
    if pars:
        smil_for[f] = smil
        smil_dur[smil] = sum(float(b) - float(a) for a, b in re.findall(r'clipBegin="([\d.]+)s" clipEnd="([\d.]+)s"', "".join(pars)))
        new_files[smil] = ('<?xml version="1.0" encoding="UTF-8"?>\n<smil xmlns="http://www.w3.org/ns/SMIL" '
                           'xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">\n'
                           f'<body><seq id="seq" epub:textref="{rel(smil, f)}">\n' + "\n".join(pars) + "\n</seq></body></smil>\n")
assert idx == len(sentences)


def hms(sec):
    return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02}:{sec % 60:06.3f}"


# --- nav: keep the book's own EPUB 3 nav, or generate one from each document's first sentence --------
nav_item = next((i for i, m in epub.manifest.items() if "nav" in m["properties"].split()), None)
if nav_item is None:
    nav_path = posixpath.join(epub.opf_dir, "nav.xhtml")
    links = []
    for f in text_docs(cfg, epub):
        first = next((s["text"] for s in sentences if s["file"] == f), None)
        if first:
            links.append(f'<li><a href="{rel(nav_path, f)}">{html.escape(first[:14] + ("…" if len(first) > 14 else ""))}</a></li>')
    new_files[nav_path] = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE html>\n<html xmlns="http://www.w3.org/1999/xhtml" '
                           f'xmlns:epub="http://www.idpf.org/2007/ops" lang="{LANG}" xml:lang="{LANG}"><head><meta charset="UTF-8"/>'
                           '<title>目次</title></head><body><nav epub:type="toc" id="toc"><h1>目次</h1><ol>'
                           + "".join(links) + "</ol></nav></body></html>\n")

# --- OPF (EPUB 3) ----------------------------------------------------------------------------------------
opf_root = epub.opf
uid_name = opf_root.get("unique-identifier")
uid = next((e.text for e in opf_root.iter("{http://purl.org/dc/elements/1.1/}identifier") if e.get("id") == uid_name), None)
creators = [e.text for e in opf_root.iter("{http://purl.org/dc/elements/1.1/}creator") if e.text]
cover_id = next((m.get("content") for m in opf_root.iter("{http://www.idpf.org/2007/opf}meta") if m.get("name") == "cover"), None)
items = []
for iid, m in epub.manifest.items():
    props = set(m["properties"].split())
    if iid == cover_id or "cover-image" in props:
        props.add("cover-image")
    extra = ""
    if m["path"] in smil_for:
        items.append(f'<item id="mo-{iid}" href="{html.escape(rel(epub.opf_path, smil_for[m["path"]]))}" media-type="application/smil+xml"/>')
        extra = f' media-overlay="mo-{iid}"'
    if "html" in m["media_type"]:
        content = new_files.get(m["path"]) or epub.read(m["path"])
        if "<svg" in content:
            props.add("svg")
        if "<script" in content:
            props.add("scripted")
    p = f' properties="{" ".join(sorted(props))}"' if props else ""
    items.append(f'<item id="{iid}" href="{html.escape(m["href"])}" media-type="{m["media_type"]}"{p}{extra}/>')
if nav_item is None:
    items.append(f'<item id="nav" href="{rel(epub.opf_path, posixpath.join(epub.opf_dir, "nav.xhtml"))}" '
                 'media-type="application/xhtml+xml" properties="nav"/>')
for tid in track_order:
    items.append(f'<item id="audio-{tid}" href="{rel(epub.opf_path, audio_dir + "/" + tid + ".m4a")}" media-type="audio/mp4"/>')
durations = "".join(f'<meta property="media:duration" refines="#mo-{iid}">{hms(smil_dur[smil_for[m["path"]]])}</meta>'
                    for iid, m in epub.manifest.items() if m["path"] in smil_for)
# the original's layout settings, as they were: page direction (vertical Japanese turns right to left), per-page
# spine attributes (linear, page-spread-*), and its EPUB 2 <meta name content> entries (primary-writing-mode ...)
ns_opf = "{http://www.idpf.org/2007/opf}"
spine_el = opf_root.find(f"{ns_opf}spine")
spine_attrs = "".join(f' {k}="{html.escape(v)}"' for k, v in spine_el.attrib.items())
itemrefs = [" ".join(f'{k}="{html.escape(v)}"' for k, v in ir.attrib.items()) for ir in spine_el]
old_metas = "".join(f'<meta name="{html.escape(m.get("name"))}" content="{html.escape(m.get("content") or "")}"/>'
                    for m in opf_root.iter(f"{ns_opf}meta") if m.get("name"))
new_files[epub.opf_path] = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="{uid_name}" xml:lang="{LANG}">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="{uid_name}">{html.escape(uid or "")}</dc:identifier>
<dc:title>{html.escape(title)}</dc:title>
{"".join(f"<dc:creator>{html.escape(c)}</dc:creator>" for c in creators)}
<dc:language>{LANG}</dc:language>
<meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
{old_metas}
<meta property="media:active-class">-epub-media-overlay-active</meta>
<meta property="media:duration">{hms(sum(smil_dur.values()))}</meta>
{durations}
</metadata>
<manifest>
{chr(10).join(items)}
</manifest>
<spine{spine_attrs}>
{chr(10).join(f'<itemref {a}/>' for a in itemrefs)}
</spine>
</package>
"""
# highlight style, appended to every stylesheet the book uses
for iid, m in epub.manifest.items():
    if m["media_type"] == "text/css":
        new_files[m["path"]] = epub.read(m["path"]) + "\n.-epub-media-overlay-active { background-color: rgba(255, 210, 80, 0.45); }\n"

print("writing the EPUB ...", flush=True)
tmp = OUT + ".part"
with zipfile.ZipFile(tmp, "w") as z:
    z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
    for info in epub.zip.infolist():
        if info.filename == "mimetype" or info.filename in new_files or info.filename.endswith("calibre_bookmarks.txt"):
            continue
        z.writestr(info, epub.zip.read(info.filename), compress_type=zipfile.ZIP_DEFLATED)
    for name, data in new_files.items():
        z.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
    for tid in track_order:
        z.write(f"{W}/epub_audio/{tid}.m4a", f"{audio_dir}/{tid}.m4a".lstrip("/"), compress_type=zipfile.ZIP_STORED)
os.replace(tmp, OUT)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1048576:.0f} MB, {hms(sum(smil_dur.values()))} of narration)")

# --- quality report (also saved as WORK/report.txt) -------------------------------------------------------------
plan_of = {r["id"]: r for r in json.load(open(f"{W}/plan.json"))["sentences"]}
lines = []


def say(text=""):
    lines.append(text)


def pct(xs, q):
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


timed = [(s, t, c) for s, t, c in zip(sentences, times, clips) if c]
say(f"quality report for {title}")
say(f"  sentences with audio: {len(timed)} of {len(sentences)}")
silent = [s for s, c in zip(sentences, clips) if not c]
if silent:
    pages = {}
    for s in silent:
        pages.setdefault(posixpath.basename(s["file"]), []).append(s["text"].strip()[:24])
    say("  without audio, by page: " + "; ".join(f"{p} ({len(v)}: {v[0]}...)" for p, v in pages.items()))
heads = [s for s in sentences if s.get("heading")]
silent_heads = [s["text"].strip() for s, c in zip(sentences, clips) if s.get("heading") and not c]
say(f"  headings with audio: {len(heads) - len(silent_heads)} of {len(heads)}"
    + (f"; silent: {silent_heads}" if silent_heads else ""))
durs = [c["end"] - c["begin"] for _, _, c in timed]
say(f"  clips empty or reversed: {sum(d <= 0 for d in durs)}, under 0.2 s: {sum(d < 0.2 for d in durs)}")
rates = sorted(len(s["norm"]) / (t["end"] - t["start"]) for s, t, _ in timed
               if len(s["norm"]) >= 10 and t["end"] > t["start"])
pace = pct(rates, 0.5)
fast = [s for s, t, _ in timed if len(s["norm"]) >= 6 and len(s["norm"]) / max(1e-6, t["end"] - t["start"]) > 3 * pace]
say(f"  reading pace (median): {pace:.1f} chars/s; sentences timed faster than 3x that: {len(fast)}"
    + (f" {[s['text'].strip()[:20] for s in fast[:5]]}" if fast else ""))
onsets = {tr: [a for a, _ in segs] for tr, segs in speech.items()}


def silence_before(tr, a):                     # from a clip's start to the first speech after it
    segs, ons = speech.get(tr, []), onsets.get(tr, [])
    i = bisect.bisect_right(ons, a) - 1
    if i >= 0 and a < segs[i][1]:
        return 0.0
    return segs[i + 1][0] - a if i + 1 < len(segs) else 0.0


lead = sorted(silence_before(c["track"], c["begin"]) for _, _, c in timed)
say(f"  silence before the narrator when a sentence is tapped: median {pct(lead, .5):.2f} s, 99% {pct(lead, .99):.2f} s,"
    f" over 0.5 s: {sum(x > 0.5 for x in lead)}")
spans = {}
for _, _, c in timed:
    spans.setdefault(c["track"], []).append((c["begin"], c["end"]))
speech_total, outside = 0.0, {}
for tr, segs in speech.items():
    cs = sorted(spans.get(tr, []))
    for a, b in segs:
        speech_total += b - a
        free = b - a - sum(max(0.0, min(b, e) - max(a, st)) for st, e in cs)
        if free > 0.05:
            outside[tr] = outside.get(tr, 0.0) + free
say(f"  narration outside every clip: {sum(outside.values()):.0f} s of {speech_total / 3600:.1f} h of speech"
    + (" (most in " + ", ".join(f"{tr} {v:.0f} s" for tr, v in sorted(outside.items(), key=lambda x: -x[1])[:3]) + ")"
       if outside else ""))
diffs = []
for s, t, _ in timed:
    r = plan_of.get(s["id"], {})
    if r.get("heard_from_start") and r.get("track") == t["track"]:
        diffs.append((abs(t["start"] - r["first_heard"]), s, t, r))
d = sorted(x[0] for x in diffs)
off = sorted((x for x in diffs if x[0] > 1.0), key=lambda x: -x[0])
say(f"  sentence start vs where Deepgram heard its first sound (sentences heard from their start): median {pct(d, .5):.2f} s, 95% {pct(d, .95):.2f} s;"
    f" more than 1 s apart: {len(off)} (Deepgram is wrong in some of these; the largest are worth a listen)")
for gap, s, t, r in off[:8]:
    say(f"      {t['track']} {t['start']:8.1f} s (Deepgram {r['first_heard']:8.1f} s)  {s['text'].strip()[:30]}")
say(f"  boundary rule: {moved} starts moved over 0.1 s to where speech resumes after the pause before them")
print("\n".join(lines))
open(f"{W}/report.txt", "w").write("\n".join(lines) + "\n")
