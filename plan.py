# /// script
# requires-python = "==3.12.*"
# dependencies = ["fugashi", "unidic-lite"]
# ///
"""Step 1c (local, called by prepare.py): locate every book sentence roughly in the audio (via Deepgram's words) and cut tracks into aligner-sized chunks.

usage: python plan.py WORK_DIR
Output plan.json:
  sentences: rough location per sentence (track, rough start/end), empty if not found in the audio
  chunks:    [{track, start, end, sentences: [ids]}], each <= MAX_CHUNK seconds, cut in pauses between sentences
"""
import bisect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from book import book_sentences, load_config  # noqa: E402
from reading import reading  # noqa: E402

cfg = load_config(os.path.abspath(sys.argv[1]))
W = cfg["work"]
MAX_CHUNK = 160.0          # the aligner's limit is 180 s
tracks = json.load(open(f"{W}/tracks.json"))
vad = json.load(open(f"{W}/vad.json"))
sentences = book_sentences(cfg)

# --- what was heard and what the book says, both as readings (katakana) ------------------------------
# Comparing readings, not spellings: the book's "１２" and "鼠" and Deepgram's "十二" and "ねずみ" read the same.
H, H_pos, WORDS = [], [], {}             # transcript reading chars; (track, start, end) of each
for t in tracks:
    words = json.load(open(f"{W}/deepgram/{t['id']}.json"))["results"]["channels"][0]["alternatives"][0]["words"]
    WORDS[t["id"]] = [(w["start"], w["end"]) for w in words]
    text, times = "", []
    for w in words:                      # each character of the track's text with its time, spread over its word
        n = len(w["word"])
        text += w["word"]
        times += [(w["start"] + i * (w["end"] - w["start"]) / n, w["start"] + (i + 1) * (w["end"] - w["start"]) / n)
                  for i in range(n)]
    for c, i in reading(text):
        H.append(c)
        H_pos.append((t["id"], *times[i]))
H = "".join(H)

R, R_sent = [], []                       # book reading chars; the sentence of each
sent_start = [0]
for k, s in enumerate(sentences):
    r = [c for c, _ in reading(s["text"])]
    R += r
    R_sent += [k] * len(r)
    sent_start.append(len(R))
R = "".join(R)

# --- anchors: K-character readings unique in the book, found in the transcript, kept if in order ---------
K = 10
grams = {}
for r in range(len(R) - K + 1):
    g = R[r:r + K]
    grams[g] = -1 if g in grams else r            # -1 marks "not unique"
pairs = [(h, grams[H[h:h + K]]) for h in range(len(H) - K + 1) if grams.get(H[h:h + K], -1) >= 0]

# longest chain with both positions increasing (patience-sort LIS on r; pairs are already sorted by h)
tails, tails_idx, parent = [], [], [None] * len(pairs)
for i, (h, r) in enumerate(pairs):
    j = bisect.bisect_left(tails, r)
    if j == len(tails):
        tails.append(r)
        tails_idx.append(i)
    else:
        tails[j], tails_idx[j] = r, i
    parent[i] = tails_idx[j - 1] if j else None
chain, i = [], tails_idx[-1] if tails_idx else None
while i is not None:
    chain.append(pairs[i])
    i = parent[i]
chain.reverse()
anchored, last_r, last_h = [], -1, -1     # (book char, transcript char) for each char of the chained anchors
for h, r in chain:
    for d in range(K):
        if r + d > last_r and h + d > last_h:
            anchored.append((r + d, h + d))
            last_r, last_h = r + d, h + d


# --- between anchors: edit-distance alignment with affine gaps ---------------------------------------------
# Every book char ends up matched (same reading heard), substituted (misheard, but placed in time) or deleted
# (not read). A gap costs more to open than to extend, so text the narrator skips (a table of contents)
# comes out as one long deletion rather than scattered matches against unrelated speech.
MATCH, MISMATCH, OPEN, EXTEND = 1.0, -1.0, -3.0, -0.5
NEG = float("-inf")


def align(a, b):
    """Global alignment of strings a and b; returns for each char of a the index into b it aligns to, or -1."""
    n, m = len(a), len(b)
    if n == 0:
        return []
    if m == 0:
        return [-1] * n
    # Gotoh: M (a[i-1] with b[j-1]), X (a[i-1] deleted), Y (b[j-1] inserted); trace keeps the best state per cell
    M = [[NEG] * (m + 1) for _ in range(n + 1)]
    X = [[NEG] * (m + 1) for _ in range(n + 1)]
    Y = [[NEG] * (m + 1) for _ in range(n + 1)]
    TM = [bytearray(m + 1) for _ in range(n + 1)]
    TX = [bytearray(m + 1) for _ in range(n + 1)]
    TY = [bytearray(m + 1) for _ in range(n + 1)]
    M[0][0] = 0.0
    for i in range(1, n + 1):
        X[i][0], TX[i][0] = OPEN + EXTEND * i, 1 if i > 1 else 0
    for j in range(1, m + 1):
        Y[0][j], TY[0][j] = OPEN + EXTEND * j, 2 if j > 1 else 0
    for i in range(1, n + 1):
        ai, Mi, Xi, Yi, Mp, Xp, Yp = a[i - 1], M[i], X[i], Y[i], M[i - 1], X[i - 1], Y[i - 1]
        TMi, TXi, TYi = TM[i], TX[i], TY[i]
        for j in range(1, m + 1):
            s = MATCH if ai == b[j - 1] else MISMATCH
            best, st = Mp[j - 1], 0
            if Xp[j - 1] > best:
                best, st = Xp[j - 1], 1
            if Yp[j - 1] > best:
                best, st = Yp[j - 1], 2
            Mi[j], TMi[j] = best + s, st
            best, st = Mp[j] + OPEN + EXTEND, 0
            if Xp[j] + EXTEND > best:
                best, st = Xp[j] + EXTEND, 1
            if Yp[j] + OPEN + EXTEND > best:
                best, st = Yp[j] + OPEN + EXTEND, 2
            Xi[j], TXi[j] = best, st
            best, st = Mi[j - 1] + OPEN + EXTEND, 0
            if Xi[j - 1] + OPEN + EXTEND > best:
                best, st = Xi[j - 1] + OPEN + EXTEND, 1
            if Yi[j - 1] + EXTEND > best:
                best, st = Yi[j - 1] + EXTEND, 2
            Yi[j], TYi[j] = best, st
    out = [-1] * n
    i, j = n, m
    state = max((M[n][m], 0), (X[n][m], 1), (Y[n][m], 2))[1]
    while i > 0 or j > 0:
        if state == 0:
            out[i - 1] = j - 1
            state = TM[i][j]
            i, j = i - 1, j - 1
        elif state == 1:
            state = TX[i][j]
            i -= 1
        else:
            state = TY[i][j]
            j -= 1
    return out


r2h = [-1] * len(R)                      # book char -> transcript char it aligns to, or -1 (not read)
prev_r, prev_h = -1, -1
for r, h in anchored + [(len(R), len(H))]:
    seg = align(R[prev_r + 1:r], H[prev_h + 1:h])
    for d, x in enumerate(seg):
        r2h[prev_r + 1 + d] = prev_h + 1 + x if x >= 0 else -1
    if r < len(R):
        r2h[r] = h
    prev_r, prev_h = r, h
heard = [r2h[r] >= 0 and R[r] == H[r2h[r]] for r in range(len(R))]
print(f"{len(pairs)} anchor hits, {len(chain)} in order; book reading {len(R)} chars, "
      f"{sum(heard)} heard ({sum(heard) / max(1, len(R)):.1%}), {sum(x >= 0 for x in r2h)} placed")

# --- each sentence ---------------------------------------------------------------------------------------
# Heard: its dominant run covers at least half of its reading. A run is a stretch of its heard characters that
# lie together in the transcript: book and transcript positions keep the same offset through matches and misheard
# words, and shift only by a few characters where the readings differ in length (年, 月, 日 added in a spoken
# date; 一色 read イッショク in the book but heard as イッシキ). Scattered matches
# (イ, ラ, ス, ト inside 「クラスメイト」 and ハチ in 八幡 half a minute later) form no run of that size.
RUN_SLACK = 3


def dominant_run(k):
    a, b = sent_start[k], sent_start[k + 1]
    runs, run, prev = [], [], None
    for r in range(a, b):
        if not heard[r]:
            continue
        h = r2h[r]
        if prev and (H_pos[h][0] != H_pos[prev[1]][0] or abs((h - prev[1]) - (r - prev[0])) > RUN_SLACK):
            runs.append(run)
            run = []
        run.append(h)
        prev = (r, h)
    runs.append(run)
    best = max(runs, key=len)
    return best if b > a and 2 * len(best) >= b - a else None


found = [dominant_run(k) for k in range(len(sentences))]
# the narrator's pace, measured: reading chars per second over sentences heard in full (median)
rates = sorted((sent_start[k + 1] - sent_start[k]) / (H_pos[x[-1]][2] - H_pos[x[0]][1])
               for k, x in enumerate(found) if x and sent_start[k + 1] - sent_start[k] >= 10
               and len(x) == sent_start[k + 1] - sent_start[k] and H_pos[x[-1]][2] > H_pos[x[0]][1])
RATE = rates[len(rates) // 2]
print(f"narrator's pace: {RATE:.1f} reading chars/s (median of {len(rates)} fully heard sentences)")

rough = [None] * len(sentences)
for k, run in enumerate(found):
    if run:
        rough[k] = {"track": H_pos[run[0]][0], "start": H_pos[run[0]][1], "end": H_pos[run[-1]][2],
                    "last_heard": H_pos[run[-1]][1],           # start of its last heard character
                    "first_heard": (H_pos[run[0]][1] + H_pos[run[0]][2]) / 2}   # middle of its first one

# Sentences Deepgram didn't catch between two heard ones on the same track (quick short lines: 「え？」,
# 「ふーん……」) are in the audio if the time between their neighbours could hold them, at no more than twice
# the narrator's pace. Text the narrator skips needs far more time than there is (a table of contents: minutes).
k = 0
while k < len(rough):
    if rough[k] is not None:
        k += 1
        continue
    j = k
    while j < len(rough) and rough[j] is None:
        j += 1
    prev, nxt = (rough[k - 1] if k else None), (rough[j] if j < len(rough) else None)
    if prev and nxt and prev["track"] == nxt["track"] and nxt["start"] >= prev["end"]:
        gap, need = nxt["start"] - prev["end"], (sent_start[j] - sent_start[k]) / (2 * RATE)
        if gap >= need:
            t0 = prev["end"]
            for i in range(k, j):                   # rough times only: share the gap by length
                t1 = t0 + gap * (sent_start[i + 1] - sent_start[i]) / max(1, sent_start[j] - sent_start[k])
                rough[i] = {"track": prev["track"], "start": t0, "end": t1, "unheard": True}
                t0 = t1
    k = j
missing = [sentences[k]["id"] for k, x in enumerate(rough) if x is None]
print(f"sentences located: {len(sentences) - len(missing)}/{len(sentences)} "
      f"({sum(1 for x in rough if x and x.get('unheard'))} of them between heard neighbours); not in audio: {len(missing)}")

# --- chunking --------------------------------------------------------------------------------------------
GAPS = {t["id"]: [(s[i][1], s[i + 1][0]) for s in [vad[t["id"]]] for i in range(len(s) - 1)] for t in tracks}


def silences(track):
    """VAD pauses, minus the speech inside them that VAD missed (quiet lines): words Deepgram heard *starting*
    inside the pause. (Deepgram's word spans run back to back, so a word that follows a pause covers part of it.)"""
    out, words = [], WORDS[track]
    starts = [w[0] for w in words]
    for a, b in GAPS[track]:
        inside = words[bisect.bisect_right(starts, a + 0.15):bisect.bisect_left(starts, b - 0.15)]
        if not inside:
            out.append((a, b))
            continue
        for x, y in [(a, inside[0][0]), (min(inside[-1][1], b), b)]:
            if y - x >= 0.2:
                out.append((x, y))
    return out


QUIET = {t["id"]: silences(t["id"]) for t in tracks}
DUR = {t["id"]: t["end"] - t["start"] for t in tracks}


def cut_before(track, t, prev_heard=None):
    """A cut point inside the pause just before speech that starts near t.
    Given when the previous sentence's last matched character starts: the first silence after it, if that comes
    before t. (Deepgram's word starts are reliable, its word ends overshoot into pauses; the next sentence's start
    can be off by its unmatched opening words; and the longest pause nearby can be a comma inside a sentence.)"""
    g = GAPS[track]
    if prev_heard is not None:
        q = QUIET[track]
        i = bisect.bisect_right([b for _, b in q], prev_heard + 0.1)
        if i < len(q) and q[i][0] < t + 0.5:
            a, b = q[i]
            return (a + b) / 2 if b - a < 2 else max(a + 0.5, (a + b) / 2 - 1.0)
    i = bisect.bisect_left([e for _, e in g], t - 1.5)
    best = None
    while i < len(g) and g[i][1] <= t + 0.6:
        if best is None or abs(g[i][1] - t) < abs(best[1] - t):
            best = g[i]
        i += 1
    return (best[0] + best[1]) / 2 if best else max(0.0, t - 0.3)


def cut_after(track, t):
    """A cut point inside the first pause after speech that ends near t."""
    for a, b in GAPS[track]:
        if b > t - 0.3 and a >= t - 1.0:
            return (a + b) / 2 if b - a < 2 else a + 0.5
    return DUR[track]


chunks = []
by_track = {}
for k, x in enumerate(rough):
    if x:
        by_track.setdefault(x["track"], []).append(k)
for track, ks in by_track.items():
    # chunk boundaries only at sentences whose rough times are in order (stray mismatches can't be boundaries),
    # and, after the first, only where both it and the sentence before were heard on this track (the cut goes in
    # the silence after the one before; a line placed between heard neighbours has only a rough time)
    ks = [k for i, k in enumerate(ks) if i == 0 or rough[k]["start"] >= rough[ks[i - 1]]["start"] - 1]
    ks = [k for i, k in enumerate(ks) if i == 0 or (not rough[k].get("unheard") and rough[k - 1]
                                                    and not rough[k - 1].get("unheard")
                                                    and rough[k - 1]["track"] == track)]
    first, g_start = ks[0], cut_before(track, rough[ks[0]]["start"])
    for i, k in enumerate(ks):
        next_start = rough[ks[i + 1]]["start"] if i + 1 < len(ks) else None
        upcoming = rough[ks[i + 2]]["start"] if i + 2 < len(ks) else rough[ks[-1]]["end"]
        if next_start is None or upcoming - g_start > MAX_CHUNK:
            end = (cut_before(track, next_start, rough[ks[i + 1] - 1]["last_heard"]) if next_start is not None
                   else cut_after(track, rough[k]["end"]))
            last = ks[i + 1] - 1 if next_start is not None else k
            # the chunk's text is every located sentence from `first` to `last` (contiguous, as spoken)
            chunks.append({"track": track, "start": round(g_start, 3), "end": round(end, 3),
                           "sentences": [sentences[j]["id"] for j in range(first, last + 1) if rough[j]]})
            if next_start is not None:
                first, g_start = ks[i + 1], end

# --- check: each chunk's text should be what is heard in its audio, and its audio should be its text ---------
idx = {s["id"]: k for k, s in enumerate(sentences)}
book_of = {r2h[r]: r for r in range(len(R)) if heard[r]}     # transcript char -> the book char heard there
heard_at = {}
for h, (tr, t0, _) in enumerate(H_pos):
    heard_at.setdefault(tr, []).append((t0, h))
report = []
for n, c in enumerate(chunks):
    ks = {idx[i] for i in c["sentences"]}
    inside = [h for t0, h in heard_at.get(c["track"], []) if c["start"] <= t0 < c["end"]]
    text = sum(sent_start[k + 1] - sent_start[k] for k in ks)
    text_ok = sum(1 for h in inside if h in book_of and R_sent[book_of[h]] in ks)
    report.append((min(text_ok / max(1, text), text_ok / max(1, len(inside))), n, text_ok / max(1, text),
                   text_ok / max(1, len(inside)), len(inside)))
report.sort()
print("chunks whose text and audio agree least (share of text heard in the chunk / share of its audio that is "
      "its text):")
for _, n, tc, ac, nh in report[:5]:
    c = chunks[n]
    print(f"  {c['track']} {c['start']:7.1f}-{c['end']:7.1f}s  text {tc:.0%}  audio {ac:.0%}  ({nh} heard chars)  "
          f"{sentences[idx[c['sentences'][0]]]['text'][:20]}")

lens = [c["end"] - c["start"] for c in chunks]
print(f"{len(chunks)} chunks, length {min(lens):.0f}-{max(lens):.0f}s, mean {sum(lens) / len(lens):.0f}s; "
      f"over 178 s: {sum(x > 178 for x in lens)}")
json.dump({"sentences": [{"id": s["id"], **(x or {})} for s, x in zip(sentences, rough)], "chunks": chunks,
           "missing": missing}, open(f"{W}/plan.json", "w"), ensure_ascii=False)
index = {s["id"]: s for s in sentences}
for t in tracks:
    cs = [c for c in chunks if c["track"] == t["id"]]
    if cs:
        print(f"  {t['id']} {t['title'][:10]:10} chunks={len(cs):3}  book text starts at {cs[0]['start']:7.1f}s: "
              f"{index[cs[0]['sentences'][0]]['text'][:20]}")
    else:
        print(f"  {t['id']} {t['title'][:10]:10} no book text found")
