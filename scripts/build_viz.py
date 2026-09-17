"""Inject a recorded training trace into the visualiser template.

The Artifact sandbox blocks the page from fetching anything, so the trace has
to ship inside the HTML rather than beside it.

    python scripts/build_viz.py training.json viz/index.html
"""

from __future__ import annotations

import sys
from pathlib import Path

PLACEHOLDER = "/*__DATA__*/ null"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    trace, out = Path(sys.argv[1]), Path(sys.argv[2])
    template = Path(__file__).resolve().parent.parent / "viz" / "template.html"

    html = template.read_text()
    if PLACEHOLDER not in html:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in {template}")

    data = trace.read_text().strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(PLACEHOLDER, data))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
