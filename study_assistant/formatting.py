"""Helpers that make model-written math render as LaTeX in Streamlit's Markdown."""

from __future__ import annotations

import re

from .llm import LATEX_JSON_ESCAPE_COLLISIONS

# The lookbehind skips "\\\\[2pt]", a LaTeX line break with spacing, which is not a delimiter.
_DISPLAY_PAREN = re.compile(r"(?<!\\)\\\[(.+?)\\\]", re.DOTALL)
_INLINE_PAREN = re.compile(r"(?<!\\)\\\((.+?)\\\)", re.DOTALL)


def markdown_math(text: str) -> str:
    """Convert \\( ... \\) and \\[ ... \\] delimiters, which Streamlit does not render, to $ and $$."""
    text = _DISPLAY_PAREN.sub(lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n", text)
    return _INLINE_PAREN.sub(lambda m: f"${m.group(1).strip()}$", text)


# "\\b", "\\f", "\\t" and "\\r" at the start of a LaTeX command become control characters
# when decoded as JSON escapes ("\\frac" -> form feed + "rac"). These are mapped back.
_DECODED_ESCAPES = {"\b": "b", "\f": "f", "\t": "t", "\r": "r"}
_CONTROL_BEFORE_WORD = re.compile(r"([\b\f\t\r])([A-Za-z]+)")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_INLINE_CODE = re.compile(r"^`+(.*?)`+$", re.DOTALL)
_PADDED_INLINE = re.compile(r"(?<![\\$])\$(?!\$)\s*([^$\n]+?)\s*\$(?!\$)")
_STRONG_MATH = re.compile(r"\\[A-Za-z]+|[\^_{}=]")
# A lone variable, a number or an operator, optionally with brackets: "u", "3", "(1,", "2)", "+".
_WEAK_MATH = re.compile(r"[-+*/<>()\[\]|,0-9.]*[A-Za-z]?[-+*/<>()\[\]|,0-9.]*")


def _restore_command(match: re.Match[str]) -> str:
    char, word = match.group(1), match.group(2)
    command = _DECODED_ESCAPES[char] + word
    # Backspace and form feed never appear in real quiz text, so always restore them.
    # A tab or carriage return is only restored when it clearly began a LaTeX command.
    if char in "\b\f" or command in LATEX_JSON_ESCAPE_COLLISIONS:
        return "\\" + command
    return match.group(0)


def _brace_tokens(text: str) -> list[str]:
    """Split on spaces, but keep anything inside {...} together as one token."""
    tokens: list[str] = []
    for word in text.split(" "):
        if tokens and tokens[-1].count("{") > tokens[-1].count("}"):
            tokens[-1] += " " + word
        else:
            tokens.append(word)
    return tokens


def wrap_bare_math(text: str) -> str:
    """Put ``$...$`` around runs of LaTeX that a model wrote without delimiters.

    "What is \\text{proj}_u v?" becomes "What is $\\text{proj}_u v$?". Ordinary words end a
    run, single letters, numbers and operators between formula pieces stay inside it.
    """
    tokens = _brace_tokens(text)
    strong = [bool(_STRONG_MATH.search(t)) for t in tokens]
    weak = [bool(t.rstrip(".,;:?!")) and bool(_WEAK_MATH.fullmatch(t.rstrip(".,;:?!"))) for t in tokens]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if not strong[i]:
            out.append(tokens[i])
            i += 1
            continue
        # Pull in operands written just before the first command ("u \\times v"), but not the
        # articles "a" and "I" or anything that ends a sentence.
        while (
            out
            and weak[i - 1]
            and out[-1] == tokens[i - 1]
            and (tokens[i - 1] not in {"a", "A", "I"} or not tokens[i].startswith("\\"))
            and not tokens[i - 1].endswith((".", ";", ":", "?", "!"))
        ):
            out.pop()
            i -= 1
        j = i  # extend the run while the next tokens are math, ending on a strong or weak piece
        while (
            j + 1 < len(tokens)
            and (strong[j + 1] or weak[j + 1])
            and not tokens[j].endswith((".", ";", ":", "?", "!"))
        ):
            j += 1
        run = " ".join(tokens[i : j + 1])
        trail = re.search(r"[.,;:?!]+$", run)
        punct = trail.group() if trail else ""
        out.append(f"${run[: len(run) - len(punct)]}${punct}")
        i = j + 1
    return " ".join(out)


def normalize_math(text: str) -> str:
    """Make model-written math render as LaTeX in Markdown instead of as raw code.

    - restores commands such as ``\\frac`` that were turned into control characters,
    - unwraps text the model put in backticks (which Markdown shows as code),
    - wraps a bare formula such as ``\\frac{a}{b}`` in ``$...$`` so it is rendered.
    """
    text = _CONTROL_BEFORE_WORD.sub(_restore_command, str(text))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
    text = markdown_math(text).strip()
    # "$ (3, 0) $" -> "$(3, 0)$": padding inside inline math can stop it from rendering.
    text = _PADDED_INLINE.sub(lambda m: f"${m.group(1)}$", text)
    was_code = False
    if (match := _INLINE_CODE.match(text)) and match.group(1).strip():
        text, was_code = match.group(1).strip(), True
    if "$" in text:
        return text  # already delimited; trust the model
    if was_code and not _LATEX_COMMAND.search(text) and re.search(r"[\^_=]", text) and " " not in text.strip():
        return f"${text}$"  # e.g. `x^2`
    if _LATEX_COMMAND.search(text):
        return wrap_bare_math(text)
    return text


