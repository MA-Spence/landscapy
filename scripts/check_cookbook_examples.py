#!/usr/bin/env python3
"""Execute marked Python examples in the cookbook Markdown files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
COOKBOOK = ROOT / "docs" / "cookbook"
TEST_BLOCK = re.compile(
    r"```python\n# cookbook: test\n(?P<code>.*?)\n```",
    flags=re.DOTALL,
)
REFERENCE_PAGE = "<!-- cookbook: reference -->"
MANUAL_PAGE = "<!-- cookbook: manual -->"


def _pages(section: str | None) -> list[Path]:
    root = COOKBOOK / section if section else COOKBOOK
    pages = sorted(root.rglob("*.md"))
    return [path for path in pages if path.name != "README.md"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        help="Cookbook directory relative to docs/cookbook (for example foundations)",
    )
    args = parser.parse_args()

    pages = _pages(args.section)
    if not pages:
        raise SystemExit("no cookbook recipe pages found")

    os.chdir(ROOT)
    executed = 0
    recipe_count = 0
    reference_count = 0
    manual_count = 0
    for page in pages:
        source = page.read_text(encoding="utf-8")
        if MANUAL_PAGE in source:
            manual_count += 1
            print(f"SKIP {page.relative_to(ROOT)} (manual tutorial)")
            continue
        blocks = [match.group("code") for match in TEST_BLOCK.finditer(source)]
        if not blocks:
            if REFERENCE_PAGE in source:
                reference_count += 1
                print(f"SKIP {page.relative_to(ROOT)} (reference page)")
                continue
            raise RuntimeError(f"{page.relative_to(ROOT)} has no executable cookbook block")
        recipe_count += 1
        for index, code in enumerate(blocks, start=1):
            label = f"{page.relative_to(ROOT)}::block-{index}"
            namespace = {"__name__": "__cookbook__", "__file__": str(page)}
            exec(compile(code, label, "exec"), namespace)
            executed += 1
            print(f"PASS {label}")

    print(
        f"Executed {executed} cookbook example(s) from {recipe_count} recipe page(s); "
        f"skipped {reference_count} reference page(s) and {manual_count} manual tutorial(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
