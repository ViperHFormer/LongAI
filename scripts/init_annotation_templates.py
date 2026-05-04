#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from longai.data.annotations import write_annotation_templates


def main() -> None:
    target = Path("data/annotations/templates")
    write_annotation_templates(target)
    print(f"annotation_templates_written -> {target}")


if __name__ == "__main__":
    main()
