"""Minimal S-expression parser for reading dune files.

Parses S-expressions into nested Python lists and strings.
Handles () grouping, quoted strings with escapes, and bare atoms.
Comments (lines starting with ;) are stripped.
"""

from typing import List, Union

SExp = Union[str, List["SExp"]]


def parse_sexp(text: str) -> List[SExp]:
    """Parse S-expressions from text into nested lists and strings.

    Returns a list of top-level expressions.
    """
    # Strip comment lines
    lines = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if not stripped.startswith(";"):
            lines.append(line)
    text = "\n".join(lines)

    tokens = _tokenize(text)
    result = []
    pos = 0
    while pos < len(tokens):
        expr, pos = _parse_one(tokens, pos)
        if expr is not None:
            result.append(expr)
    return result


def _tokenize(text: str) -> List[str]:
    """Split text into tokens: '(', ')', and strings (quoted or bare)."""
    tokens = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in " \t\n\r":
            i += 1
        elif c == "(":
            tokens.append("(")
            i += 1
        elif c == ")":
            tokens.append(")")
            i += 1
        elif c == ";":
            # Skip rest of line (inline comment)
            while i < len(text) and text[i] != "\n":
                i += 1
        elif c == '"':
            # Quoted string
            parts = []
            i += 1
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    parts.append(text[i + 1])
                    i += 2
                elif text[i] == '"':
                    i += 1
                    break
                else:
                    parts.append(text[i])
                    i += 1
            tokens.append("".join(parts))
        else:
            # Bare atom
            start = i
            while i < len(text) and text[i] not in " \t\n\r()\";":
                i += 1
            tokens.append(text[start:i])
    return tokens


def _parse_one(tokens: List[str], pos: int) -> tuple:
    """Parse one expression starting at pos. Returns (expr, new_pos)."""
    if pos >= len(tokens):
        return None, pos
    tok = tokens[pos]
    if tok == "(":
        children = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            child, pos = _parse_one(tokens, pos)
            if child is not None:
                children.append(child)
        if pos < len(tokens):
            pos += 1  # skip ')'
        return children, pos
    elif tok == ")":
        # Unmatched close paren — skip
        return None, pos + 1
    else:
        return tok, pos + 1


def find_stanzas(sexp_list: List[SExp], name: str) -> List[List[SExp]]:
    """Return all top-level lists whose first element equals name."""
    results = []
    for item in sexp_list:
        if isinstance(item, list) and len(item) > 0 and item[0] == name:
            results.append(item)
    return results
