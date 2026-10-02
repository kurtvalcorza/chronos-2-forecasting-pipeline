"""Write carried notebook strings as short implicitly concatenated literals.

A capstone notebook carries files and archives as Python string literals. Written with a
plain ``repr`` each becomes one notebook line of up to ~1.4 million characters, which can
make a hosted notebook editor unresponsive. These helpers write the same value as a
parenthesised run of short string pieces: Python joins them back into identical text, so
every file written at run time, and its SHA-256, is unchanged.
"""

from __future__ import annotations

PIECE = 1000


def pieces(text: str, size: int = PIECE) -> list[str]:
    """Split at source line boundaries, cutting any line longer than ``size`` characters."""
    parts = []
    for line in text.splitlines(keepends=True):
        parts.extend(line[start : start + size] for start in range(0, len(line), size))
    return parts or [""]


def carried_literal(text: str, level: int = 0) -> str:
    """Return a Python literal for ``text`` whose source lines stay short."""
    parts = pieces(text)
    if len(parts) == 1:
        return repr(parts[0])
    inner = "    " * (level + 1)
    body = "".join(f"{inner}{part!r}\n" for part in parts)
    return "(\n" + body + "    " * level + ")"


def carried_dict(mapping: dict[str, str]) -> str:
    """Return a dict literal with each string value written by ``carried_literal``."""
    body = "".join(f"    {key!r}: {carried_literal(value, 1)},\n" for key, value in mapping.items())
    return "{\n" + body + "}"
