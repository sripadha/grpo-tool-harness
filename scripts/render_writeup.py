"""Render WRITEUP.md to a PDF and print the page count and the width of the figure.

Page settings: A4 portrait, 2 cm margins, Helvetica 10 pt body text
(line height 1.2), 14 pt title, 11 pt section heads, 8 pt tables
(7.5 pt table heads), Courier 8.5 pt code; the column widths of the two
tables of WRITEUP.md are in HEAD_WIDTHS.

The figure is drawn --width wide (default 11.25 cm). The PDF library draws
an image at 0.75 of the width that the HTML asks for, so the script asks
for more and then reads the drawn width from the PDF itself. It fails
(exit code 1) if the drawn width is not the width asked for, if the PDF
has more than --max-pages pages (default 2), or if its document
information names an author, a title, a subject or keywords.

Packages: markdown 3.11.3 and xhtml2pdf 0.2.28, with the pypdf and reportlab that it installs (they are
not in the project's requirements; install them in a separate
environment).

Usage:
    python scripts/render_writeup.py [--src WRITEUP.md] [--out WRITEUP.pdf]
        [--width 11.25cm] [--max-pages 2]

It prints "errors E pages N figure W cm wide". Relative paths in --src
(the figure) are read from the top of the repository.
"""
import argparse
import os
import re

import sys

import markdown
from pypdf import PdfReader
from pypdf.generic import ContentStream
from reportlab import rl_config
from xhtml2pdf import pisa

POINTS_PER_CM = 72 / 2.54
LIBRARY_IMAGE_SCALE = 0.75      # xhtml2pdf 0.2.28 draws an image at this share of the width asked for


def drawn_image_widths_cm(pdf_path):
    """The width, in cm, of every image that the PDF draws (read from the page contents)."""
    widths = []
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        last_scale = None
        for operands, operator in ContentStream(page.get_contents(), reader).operations:
            if operator == b"cm":
                last_scale = float(operands[0])
            elif operator == b"Do" and last_scale is not None:
                widths.append(last_scale / POINTS_PER_CM)
    return widths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSS = """
@page { size: a4 portrait; margin: 2cm; }
body { font-family: Helvetica; font-size: 10pt; line-height: 1.2; }
h1 { font-size: 14pt; margin: 0 0 4pt 0; }
h2 { font-size: 11pt; margin: 4pt 0 1pt 0; }
p, li { margin: 0 0 4pt 0; }
table { border: 0.5pt solid #888; font-size: 8pt; }
th { font-size: 7.5pt; }
td, th { padding: 1pt 2pt; border: 0.5pt solid #888; vertical-align: top; }
code { font-family: Courier; font-size: 8.5pt; }
"""

# Column widths, set by the text of a table head (a head not named here gets the default width).
HEAD_WIDTHS = {
    "Item": "14%",                                        # the setup table
    "Arm": "10%", "Dev pass % [CI]": "15%", "Dev difference": "18%",     # the results table
    "Test pass % [CI]": "15%", "Test difference": "18%", "All 8 equal, %": "12%", "Malformed, %": "12%",
}


def main():
    ap = argparse.ArgumentParser(description="Render WRITEUP.md to a PDF; print the page count and the figure width.")
    ap.add_argument("--src", default=os.path.join(ROOT, "WRITEUP.md"))
    ap.add_argument("--out", default=os.path.join(ROOT, "WRITEUP.pdf"))
    ap.add_argument("--width", default="11.25cm", help="width of the figure on the page, in cm")
    ap.add_argument("--max-pages", type=int, default=2, help="fail if the PDF has more pages than this")
    a = ap.parse_args()
    with open(a.src, encoding="utf-8") as f:
        body = markdown.markdown(f.read(), extensions=["tables"])
    width_cm = float(a.width.replace("cm", ""))
    body = body.replace("<img ", f'<img width="{width_cm / LIBRARY_IMAGE_SCALE:.3f}cm" ')
    for head, width in HEAD_WIDTHS.items():
        body = body.replace(f"<th>{head}</th>", f'<th width="{width}">{head}</th>')
    html = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")

    def link(uri, rel):
        p = os.path.join(ROOT, uri)
        return p if os.path.exists(p) else uri

    rl_config.useA85 = 0        # binary streams: scripts/make_package.sh then takes the PDF for a binary file
    with open(a.out, "wb") as f:
        r = pisa.CreatePDF(html, dest=f, link_callback=link)
    with open(a.out, "rb") as f:
        pages = len(re.findall(rb"/Type\s*/Page[^s]", f.read()))
    widths = drawn_image_widths_cm(a.out)
    info = PdfReader(a.out).metadata or {}
    named = {k: v for k, v in info.items() if k in ("/Author", "/Title", "/Subject", "/Keywords") and v}
    shown = ", ".join(f"{w:.2f}" for w in widths) if widths else "none"
    print("errors", r.err, "pages", pages, "figure", shown, "cm wide")
    problems = []
    if r.err:
        problems.append("the PDF library reported errors")
    if pages > a.max_pages:
        problems.append(f"{pages} pages, more than {a.max_pages}")
    if "<img " in body and (len(widths) != 1 or abs(widths[0] - width_cm) > 0.05):
        problems.append(f"the figure was drawn {shown} cm wide, not {width_cm:.2f} cm")
    if named:
        problems.append(f"the PDF names {sorted(named)} in its document information")
    if problems:
        print("FAILED:", "; ".join(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
