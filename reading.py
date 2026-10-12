"""Japanese text -> katakana reading, each reading character tied to a source character index."""
import re
import unicodedata

import fugashi

from book import kanji_number

_tagger = fugashi.Tagger()


def normalize(text):
    """NFKC (full-width, circled numbers, Kangxi radicals), then digit runs as kanji numerals.
    Returns (normalized text, source index of each normalized character)."""
    chars, src = [], []
    for i, c in enumerate(text):
        for d in unicodedata.normalize("NFKC", c):
            chars.append(d)
            src.append(i)
    s, out, out_src, k = "".join(chars), [], [], 0
    for m in re.finditer(r"\d+", s):
        out += chars[k:m.start()]
        out_src += src[k:m.start()]
        num = kanji_number(m.group())
        out += list(num)
        out_src += [src[m.start()]] * len(num)
        k = m.end()
    out += chars[k:]
    out_src += src[k:]
    return "".join(out), out_src


def to_kata(s):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def reading(text):
    """[(katakana char, source index in text)] for the spoken characters of text."""
    s, src = normalize(text)
    out, pos = [], 0
    for w in _tagger(s):
        surf = w.surface
        start = s.find(surf, pos) if surf else pos
        if start < 0:
            start = pos
        pos = start + len(surf)
        kana = getattr(w.feature, "kana", None)
        r = to_kata(kana if kana and kana != "*" else surf).lower()
        r = re.sub(r"[^\w]", "", r)
        if not r or not surf:
            continue
        # spread the token's reading over its source characters
        for j, ch in enumerate(r):
            out.append((ch, src[start + min(len(surf) - 1, j * len(surf) // len(r))]))
    return out
