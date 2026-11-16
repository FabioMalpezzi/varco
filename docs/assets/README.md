# Diagrams

The seven diagrams of the documentation, each as two SVG files: one for
light pages and one for dark pages.

| Diagram | Shows | Meant for |
|---|---|---|
| `overview` | The program stays in the container; the computing happens on the Mac's GPU | `README.md` |
| `two-bridges` | The MLX bridge and the PyTorch bridge, end to end | `docs/architecture.md` |
| `operation-path` | One operation from the program to the GPU and back, for each bridge | `docs/architecture.md` |
| `channels` | The channel with Docker Desktop and with Kiri | `docs/architecture.md`, `docs/setup.md` |
| `several-containers` | Several containers on the same server, for each bridge | `docs/use-cases/several-containers.md`, `docs/architecture.md` |
| `weights` | Weights read by the Mac instead of uploaded | `docs/architecture.md` |
| `urgent-reads` | A small read answered ahead of the server's queue | `docs/architecture.md` |

The logo is drawn by hand, not generated: `logo-light.svg` and
`logo-dark.svg` are the mark with the word, for light and dark pages (at
the top of `README.md`), and `logo-mark.svg` is the mark alone on a dark
tile. The mark is a V that passes through a gap in a barrier; the letters
are paths, so no font is needed to show them.

## Files

```
<name>-light.svg      the diagram on a light card
<name>-dark.svg       the same diagram on a dark card
src/<name>.html       the source: one page with the diagram as inline SVG
src/<name>-dark.html  derived from <name>.html by the build, not edited by hand
src/palette-light.css the colours of the light variant
src/palette-dark.css  the colours of the dark variant
src/build.py          derives the dark sources and exports the SVG files
```

The SVG files are generated. To change a diagram, edit `src/<name>.html`
(or a palette) and run the build.

## Embedding

GitHub shows the variant that matches the reader's theme:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/overview-dark.svg">
  <img src="docs/assets/overview-light.svg" alt="..." width="100%">
</picture>
```

The `<desc>` at the top of each SVG is a description in words that can be
shortened into the `alt` text.

## Regenerating

The diagrams are made with [diagram-design](https://github.com/cathrynlavery/diagram-design),
version 2.6.132 (MIT), in its default skin with larger type and system
fonts. The build needs Python 3 and two scripts of that project, found
under `skills/diagram-design/scripts/` in its repository: `export_svg.py`
and `self_check.py`. It needs no browser and no network.

```sh
python3 docs/assets/src/build.py <path to>/skills/diagram-design
```

For each `src/<name>.html` the build writes the light palette into the
page, derives `src/<name>-dark.html` (the same page with the dark palette),
exports both with `export_svg.py --system-fonts`, runs
`self_check.py --offline` on both pages, and checks that each SVG is
well-formed and loads nothing from outside.

To look at the result as GitHub shows it in the README column, which is
about 830 px wide:

```sh
rsvg-convert -w 830 docs/assets/overview-light.svg -o /tmp/overview-light.png
```

## Conventions

The seven diagrams are one family. A new one should keep to this:

- **Size.** The `viewBox` is 880 wide; the height follows the content.
  Names are 15 px (14 px for parts and program names), other text 12 px,
  the small capitals 11 px. Nothing is smaller, so that at 830 px the
  smallest text is still 10 px.
- **Fonts.** System fonts only (`system-ui` and `ui-monospace` stacks).
  An SVG shown through `<img>` cannot load a font. The monospaced face is
  for what is typed or sent: program names, addresses, options, messages.
- **Place.** The Linux virtual machine is on the left, macOS on the right.
- **Shapes.**

  | Thing | Shape |
  |---|---|
  | Linux virtual machine | region with a dashed border |
  | macOS | region with a solid border |
  | container | box with a header strip |
  | program | tinted box, thin border |
  | container end, Mac end (server, session) | plain box, strong border |
  | part of a server (a thread), Kiri's own programs | box with a thin grey border |
  | Apple GPU | filled box with cut corners |
  | file | sheet with a folded corner; dashed when it is a mount |
  | channel | band in the accent colour between the two regions |

- **Accent.** The accent colour marks the channel, where the bridge crosses
  from the virtual machine to macOS, and nothing else.
- **Lines.** A solid arrow is something sent or read; a dashed arrow is
  something that comes back, or a mount; a thick arrow is the path the
  diagram is about; a dotted arrow is a fallback.
- **Geometry.** Positions and sizes are multiples of 4; connectors are
  horizontal or vertical with rounded corners.
- **Colours.** Every fill is an opaque colour and the card is opaque,
  because GitHub has several light and dark themes. The colours are only
  in the two palette files.
