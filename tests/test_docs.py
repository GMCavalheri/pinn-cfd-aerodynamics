r"""Guard against Markdown constructs that GitHub's renderer corrupts inside inline math.

GitHub runs its Markdown pass over inline ``$...$`` before MathJax sees it: ``\,`` and ``\;``
lose their backslash, a ``\\`` row break collapses to ``\``, and ``_`` right after ``}``,
``)`` or ``]`` can open an emphasis span. Display math in ```math fences is safe (CLAUDE.md,
"Docs").
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted(p for p in [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md")] if "PLAN" not in p.name)
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
CODE = re.compile(r"`[^`\n]*`")
INLINE = re.compile(r"(?<![\\$])\$(?!\$)([^$\n]+?)\$")
BAD = [(r"\\[,;]", "thin space"), (r"\\\\", "row break"), (r"[})\]]_", "emphasis after bracket")]


def inline_math(text):
    text = CODE.sub("", FENCE.sub("", text))
    return [m.group(1) for m in INLINE.finditer(text)]


@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_inline_math_survives_github(path):
    problems = [
        f"{what}: ${span}$"
        for span in inline_math(path.read_text())
        for pattern, what in BAD
        if re.search(pattern, span)
    ]
    assert not problems, "\n".join(problems)
