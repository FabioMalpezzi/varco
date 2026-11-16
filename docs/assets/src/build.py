#!/usr/bin/env python3
"""Rebuild the diagrams of docs/assets from their HTML sources.

    python3 docs/assets/src/build.py <diagram-design directory>

For every <name>.html in this directory it

1. writes palette-light.css into the palette block of <name>.html, so that the
   colours have one source;
2. derives <name>-dark.html: the same file with palette-dark.css in the
   palette block and "-dark" in the ids of <title> and <desc>;
3. exports ../<name>-light.svg and ../<name>-dark.svg with that project's
   scripts/export_svg.py --system-fonts (no web font, no network request);
4. runs its scripts/self_check.py --offline on both sources, then
   checks that each SVG is well-formed XML with nothing external in it.

Only the Python standard library is needed, plus those two scripts of diagram-design
(https://github.com/cathrynlavery/diagram-design, skills/diagram-design).
"""
import re
import subprocess
import sys
import xml.dom.minidom
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
BLOCK = re.compile(r"[ \t]*/\* palette:start.*?/\* palette:end \*/", re.DOTALL)
# Written in two pieces so that a search of this directory for them finds only real cases.
FORBIDDEN = tuple(a + b for a, b in (("<", "script"), ("@", "import"), ("<", "link"), ("<", "image"),
                                     ("url(", "http"), ("/", "Users/"), ("/", "home/")))
NAMESPACES = ("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xhtml")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    skill = Path(sys.argv[1]).expanduser()
    export, check = skill / "scripts" / "export_svg.py", skill / "scripts" / "self_check.py"
    for script in (export, check):
        if not script.is_file():
            print(f"not found: {script}")
            return 2
    light = (SRC / "palette-light.css").read_text(encoding="utf-8").rstrip("\n")
    dark = (SRC / "palette-dark.css").read_text(encoding="utf-8").rstrip("\n")
    failed = False
    names = sorted(p.stem for p in SRC.glob("*.html") if not p.stem.endswith("-dark"))
    for name in names:
        source = SRC / f"{name}.html"
        text = source.read_text(encoding="utf-8")
        if len(BLOCK.findall(text)) != 1:
            print(f"{source.name}: expected exactly one palette block")
            return 1
        with_light = BLOCK.sub(lambda _: light, text)
        if with_light != text:
            source.write_text(with_light, encoding="utf-8")
        with_dark = BLOCK.sub(lambda _: dark, with_light)
        for part in ("title", "desc"):
            with_dark = with_dark.replace(f"{name}-{part}", f"{name}-dark-{part}")
        dark_source = SRC / f"{name}-dark.html"
        dark_source.write_text(with_dark, encoding="utf-8")

        for html, svg in ((source, OUT / f"{name}-light.svg"), (dark_source, OUT / f"{name}-dark.svg")):
            subprocess.run([sys.executable, str(export), str(html), str(svg), "--system-fonts"], check=True)
            if subprocess.run([sys.executable, str(check), "--offline", str(html)]).returncode != 0:
                failed = True
            body = svg.read_text(encoding="utf-8")
            xml.dom.minidom.parseString(body.encode("utf-8"))
            found = [word for word in FORBIDDEN if word in body]
            for url in re.findall(r"https?://[^\s\"'<>)]+", body):
                if url not in NAMESPACES:
                    found.append(url)
            if re.search(r"<svg\b[^>]*\s(width|height)=", body):
                found.append("a fixed width or height on <svg>")
            if found:
                print(f"{svg.name}: not allowed in an SVG shown through <img>: {found}")
                failed = True
    print(f"{len(names)} diagrams, {2 * len(names)} SVG files in {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
