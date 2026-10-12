"""Shared helpers: project config, reading any EPUB's spine, Japanese sentence splitting, text normalisation."""
import json
import os
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from html import unescape

OPF_NS = {"opf": "http://www.idpf.org/2007/opf", "dc": "http://purl.org/dc/elements/1.1/"}
LANG_CODES = {"Japanese": "ja", "Chinese": "zh", "English": "en", "Korean": "ko"}

# paragraphs and headings (chapter and section titles are read aloud too)
PARA = re.compile(r"<(?P<tag>p|h[1-6])\b[^>]*>(?P<inner>.*?)</(?P=tag)>", re.S)
# A sentence ends at 。！？ (plus any closing brackets right after), or between back-to-back quotes 」「.
# A quote followed by its tag (「筋合？」と僕は言った。) stays one sentence.
SENT_END = re.compile(r"(?:[。！？!?]++[」』）)]*+|[」』](?=[「『]))(?!と|って)")


def load_config(work):
    """book.json in the work directory: epub, audio (in order), language, optional title/spine."""
    cfg = json.load(open(os.path.join(work, "book.json")))
    cfg.setdefault("language", "Japanese")
    cfg["work"] = work
    cfg["epub_path"] = cfg["epub"] if os.path.isabs(cfg["epub"]) else os.path.join(work, cfg["epub"])
    return cfg


def norm(s):
    """Characters that are actually spoken: drop punctuation, symbols and whitespace."""
    return re.sub(r"[^\w]", "", s)


def split_sentences(text):
    out, start = [], 0
    for m in SENT_END.finditer(text):
        out.append(text[start:m.end()])
        start = m.end()
    if text[start:].strip():
        out.append(text[start:])
    merged = []                          # glue pieces with nothing to say (a lone "」") onto the previous one
    for s in out:
        if merged and not norm(s):
            merged[-1] += s
        else:
            merged.append(s)
    return merged


VOID = {"br", "img", "hr", "wbr", "input", "meta", "link"}
SILENT = {"rt", "rp"}                    # furigana readings: shown, but not part of the spoken text


def split_paragraph(inner):
    """Split a paragraph's inner HTML into sentences: [(html_piece, spoken_text)].

    Cuts only happen outside any element, so every piece is well-formed HTML; <rt>/<rp> text is not spoken.
    """
    spoken, cuts = [], [(0, 0)]          # cuts: (html offset, spoken chars before it) at nesting depth 0
    pos = depth = silent = 0
    for part in re.split(r"(<[^>]+>)", inner):
        if part.startswith("<"):
            name = (re.match(r"</?\s*([A-Za-z0-9]+)", part) or [None, ""])[1].lower()
            closing, single = part.startswith("</"), part.endswith("/>") or name in VOID
            if not single:
                depth += -1 if closing else 1
                if name in SILENT:
                    silent += -1 if closing else 1
            pos += len(part)
            if depth == 0:
                cuts.append((pos, len(spoken)))
            continue
        for tok in re.findall(r"&#?\w+;|.", part, re.S):
            pos += len(tok)
            if not silent:
                spoken.append(unescape(tok))
            if depth == 0:
                cuts.append((pos, len(spoken)))
    text = "".join(spoken)
    if not norm(text):
        return []
    out, h0, s0, boundary = [], 0, 0, 0
    for sentence in split_sentences(text):
        boundary += len(sentence)
        h1, s1 = next(((h, s) for h, s in cuts if s >= boundary and h >= h0), (len(inner), len(text)))
        if s1 > s0:
            out.append((inner[h0:h1], text[s0:s1]))
            h0, s0 = h1, s1
    if h0 < len(inner) and out:          # trailing markup with no speech joins the last sentence
        out[-1] = (out[-1][0] + inner[h0:], out[-1][1])
    return out


def paragraphs(html):
    """Yield (match, [(html_piece, spoken_text)]) for every <p> or <h1>-<h6> that has spoken text."""
    for m in PARA.finditer(html):
        pieces = split_paragraph(m.group("inner"))
        if pieces:
            yield m, pieces


class Epub:
    """Package structure of an EPUB 2/3: OPF location, manifest, spine documents (as zip paths)."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        container = ET.fromstring(self.zip.read("META-INF/container.xml"))
        self.opf_path = container.find(".//{*}rootfile").get("full-path")
        self.opf_dir = posixpath.dirname(self.opf_path)
        self.opf_text = self.zip.read(self.opf_path).decode("utf-8")
        opf = ET.fromstring(self.opf_text)
        self.opf = opf
        self.manifest = {}
        for it in opf.find("opf:manifest", OPF_NS):
            self.manifest[it.get("id")] = {"href": it.get("href"), "media_type": it.get("media-type"),
                                           "properties": it.get("properties", ""), "path": self.resolve(it.get("href"))}
        spine = opf.find("opf:spine", OPF_NS)
        self.spine_toc = spine.get("toc")
        self.spine = [ir.get("idref") for ir in spine]
        self.spine_docs = [self.manifest[i]["path"] for i in self.spine if "html" in self.manifest[i]["media_type"]]

    def resolve(self, href):
        return posixpath.normpath(posixpath.join(self.opf_dir, unescape(href).split("#")[0]))

    def read(self, path):
        return self.zip.read(path).decode("utf-8")


def track_wav(cfg, track_id):
    """16 kHz mono WAV of a track (decoded from the bundled Opus file on first use)."""
    import subprocess
    wav = os.path.join(cfg["work"], "tracks", f"{track_id}.wav")
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav[:-4] + ".ogg", "-ac", "1", "-ar", "16000",
                        wav + ".tmp.wav"], check=True)
        os.rename(wav + ".tmp.wav", wav)
    return wav


def toc_docs(epub):
    """Table-of-contents pages, as the book itself declares them (EPUB 3 nav, EPUB 2 guide type="toc").
    Narrators don't read them; an undeclared one comes out of plan.py's alignment as text that isn't heard."""
    out = {m["path"] for m in epub.manifest.values() if "nav" in m["properties"].split()}
    out |= {epub.resolve(r.get("href")) for r in epub.opf.iter("{http://www.idpf.org/2007/opf}reference")
            if r.get("type") == "toc"}
    return out


def text_docs(cfg, epub):
    """Documents whose text is aligned: the spine (or book.json "spine"), minus table-of-contents pages."""
    if cfg.get("spine"):
        return cfg["spine"]
    toc = toc_docs(epub)
    return [f for f in epub.spine_docs if f not in toc]


def book_sentences(cfg):
    """All sentences in reading order: dicts with id, file (zip path), text, norm."""
    epub = Epub(cfg["epub_path"])
    out = []
    for f in text_docs(cfg, epub):
        for m, pieces in paragraphs(epub.read(f)):
            for _, text in pieces:
                out.append({"id": f"s{len(out):05}", "file": f, "text": text, "norm": norm(text)})
    return out
