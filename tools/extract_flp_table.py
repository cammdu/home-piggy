"""Reproduce homepiggy/data/nl/financieringslast_2026_non_aow.csv from the official PDF.

Source: Staatscourant 2025 nr. 36471, "Wijzigingsregeling hypothecair krediet 2026"
        https://zoek.officielebekendmakingen.nl/stcrt-2025-36471.pdf
        (sha256 bc3cde8f0ed6aa0b68aa7ebf4c6cf888a30b256104e38a4af52faff023da47ea, retrieved 2026-11-10)

Usage:  python tools/extract_flp_table.py <downloaded.pdf> <out.csv>

Standard library only. The PDF's content streams are Flate-compressed and use
WinAnsi-encoded text, so the table is read straight from the text-drawing
operators: no OCR and no hand transcription. Character code 0x02 is the euro
sign (EuroSans font) and is dropped. The script fails loudly if the table
shape is not exactly as expected.
"""
import csv
import hashlib
import re
import sys
import zlib

EXPECTED_PDF_SHA256 = "bc3cde8f0ed6aa0b68aa7ebf4c6cf888a30b256104e38a4af52faff023da47ea"
data = open(sys.argv[1], "rb").read()
if hashlib.sha256(data).hexdigest() != EXPECTED_PDF_SHA256:
    print("WARNING: PDF hash differs from the version used to build the CSV", file=sys.stderr)
streams = []
for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
    try: streams.append(zlib.decompress(m.group(1)))
    except Exception: pass
tok = re.compile(rb"\((?:\\.|[^\\)])*\)|\[(?:\((?:\\.|[^\\)])*\)|[^\]])*\]|-?\d*\.?\d+|/[^\s/\[\]()<>]+|[A-Za-z'\"*]+")
def unesc(b):
    b = b[1:-1]
    b = re.sub(rb"\\([nrtbf()\\])", lambda m: {b'n':b'\n',b'r':b'',b't':b'\t',b'b':b'',b'f':b'',b'(':b'(',b')':b')',b'\\':b'\\'}[m.group(1)], b)
    b = re.sub(rb"\\(\d{1,3})", lambda m: bytes([int(m.group(1),8)]), b)
    return b.decode("cp1252", "replace")
out = []
for si, s in enumerate(streams):
    if b"Tj" not in s and b"TJ" not in s: continue
    items = []; x = y = 0.0; lx = ly = 0.0; stack = []
    for t in tok.findall(s):
        if t in (b"Tm",):
            x, y = float(stack[-2]), float(stack[-1]); lx, ly = x, y; stack = []
        elif t in (b"Td", b"TD"):
            lx += float(stack[-2]); ly += float(stack[-1]); x, y = lx, ly; stack = []
        elif t == b"T*":
            stack = []
        elif t == b"Tj":
            items.append((round(y,1), x, unesc(stack[-1]))); stack = []
        elif t == b"TJ":
            arr = stack[-1]; txt = "".join(unesc(p) for p in re.findall(rb"\((?:\\.|[^\\)])*\)", arr))
            items.append((round(y,1), x, txt)); stack = []
        elif t[:1].isalpha() and t not in (b"Tf",):
            stack = [] if t not in (b"Tf",) else stack
        else:
            stack.append(t)
    rows = {}
    for yy, xx, txt in items: rows.setdefault(yy, []).append((xx, txt))
    out.append(f"\n=== STREAM {si} ===")
    for yy in sorted(rows, reverse=True):
        out.append(" | ".join(t for _, t in sorted(rows[yy])))
text = "\n".join(out)
start, end = text.index("Tabel 1."), text.index("Tabel 2.")
rows = []
for line in text[start:end].splitlines():
    cells = [c.replace("\x02", "").strip() for c in line.split("|")]
    cells = [c for c in cells if c]
    if len(cells) == 13 and all(re.fullmatch(r"\d{1,2},\d%", c) for c in cells[1:]):
        inc = 0 if cells[0] == "–" else int(cells[0].replace(".", ""))
        rows.append([inc] + [c.replace("%", "").replace(",", ".") for c in cells[1:]])
assert rows[0][0] == 0 and [r[0] for r in rows[1:]] == list(range(30000, 126000, 1000)), "unexpected table rows"
cols = ["toetsinkomen_from_eur", "r_le_1500", "r_1501_2000", "r_2001_2500", "r_2501_3000", "r_3001_3500",
        "r_3501_4000", "r_4001_4500", "r_4501_5000", "r_5001_5500", "r_5501_6000", "r_6001_6500", "r_ge_6501"]
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    w.writerows(rows)
print(len(rows), "rows written;", hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest())
