#!/usr/bin/env python3
"""Check the links between the Markdown files of the repository.

For every [text](target) in a .md file: a relative target must be a file or a
directory of the repository, and a #fragment must be a heading of the file it
points to (as GitHub makes anchors). The same for the images named by the
src and srcset of an <img> or a <picture>. Links to the web are listed with
--external and not fetched.
    scripts/check-links.py [--external]
"""
import os
import re
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "*.md"], cwd=root, capture_output=True, text=True, check=True).stdout.split("\n")
files = [f for f in files if f]
link = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# The images of an <img> or a <picture>: src="..." and srcset="...".
image = re.compile(r"\b(?:src|srcset)=\"([^\"\s]+)\"")


def anchors(path):
    found, seen, fenced = set(), {}, False
    for line in open(path, encoding="utf-8"):
        if line.startswith("```"):
            fenced = not fenced
        if fenced or not line.startswith("#"):
            continue
        title = line.lstrip("#").strip()
        slug = re.sub(r"[^\w\- ]", "", title.lower().replace("`", "")).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        found.add(slug if count == 0 else f"{slug}-{count}")
    return found


bad, external, checked = [], set(), 0
for name in files:
    path = os.path.join(root, name)
    fenced = False
    for number, line in enumerate(open(path, encoding="utf-8"), 1):
        if line.startswith("```"):
            fenced = not fenced
        if fenced:
            continue
        for target in link.findall(line) + image.findall(line):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                external.add(target)
                continue
            checked += 1
            file, _, fragment = target.partition("#")
            to = os.path.normpath(os.path.join(os.path.dirname(path), file)) if file else path
            if not os.path.exists(to):
                bad.append(f"{name}:{number}: {target}: no such file")
            elif fragment and to.endswith(".md") and fragment not in anchors(to):
                bad.append(f"{name}:{number}: {target}: no such heading")
for line in bad:
    print(line)
if "--external" in sys.argv:
    for target in sorted(external):
        print(target)
print(f"{len(files)} Markdown files, {checked} links between them checked, {len(bad)} broken; {len(external)} links to the web not fetched")
sys.exit(1 if bad else 0)
