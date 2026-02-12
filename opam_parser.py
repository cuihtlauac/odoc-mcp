"""Parser for opam package files.

Converts the opam file format into a JSON-friendly dict.
"""

from typing import Any, Dict, List, Optional


def _parse_string(text: str, pos: int) -> tuple:
    """Parse a quoted string starting at pos (which points to the opening quote).

    Handles escaped quotes and triple-quoted strings.
    Returns (parsed_string, new_pos).
    """
    # Check for triple-quoted string
    if text[pos:pos + 3] == '"""':
        end = text.find('"""', pos + 3)
        if end == -1:
            return text[pos + 3:], len(text)
        return text[pos + 3:end], end + 3

    # Regular quoted string
    result = []
    i = pos + 1
    while i < len(text):
        c = text[i]
        if c == '\\' and i + 1 < len(text):
            result.append(text[i + 1])
            i += 2
        elif c == '"':
            return ''.join(result), i + 1
        else:
            result.append(c)
            i += 1
    return ''.join(result), i


def _parse_list(text: str, pos: int) -> tuple:
    """Parse a bracketed list starting at pos (which points to '[').

    Returns (list_of_strings, new_pos).
    """
    items = []
    i = pos + 1
    while i < len(text):
        c = text[i]
        if c == ']':
            return items, i + 1
        elif c == '"':
            s, i = _parse_string(text, i)
            items.append(s)
        elif c in ' \t\n\r':
            i += 1
        else:
            i += 1
    return items, i


def _parse_dependency(dep_str: str) -> Dict[str, Optional[str]]:
    """Parse a single dependency string like '"ocaml" {>= "4.08"}'.

    Returns {"package": name, "constraint": constraint_or_None}.
    """
    dep_str = dep_str.strip()
    # The dependency string from our list parser is already unquoted for the
    # package name, but might contain a constraint in braces.
    # However, in the list we get raw items that look like:
    #   ocaml   or   ocaml" {>= "4.08  (partial)
    # Actually, from our _parse_list we get the content between quotes,
    # so each item is just a package name like "ocaml".
    # The constraint follows *outside* the quotes in the original text.
    # We need a different approach - parse deps from raw text.
    return {"package": dep_str, "constraint": None}


_KNOWN_FLAGS = {"build", "with-test", "with-doc", "with-dev-setup"}


def _parse_constraint(raw: str) -> dict:
    """Split a raw constraint string into version constraint and flags.

    Takes the brace content after quote removal (e.g. "build & >= 1.1.0")
    and returns {"constraint": "...", "flags": [...]}.
    """
    if not raw:
        return {"constraint": None, "flags": []}

    # Split on '&' at top level (skip '&' inside parentheses)
    parts = []
    current = []
    depth = 0
    for ch in raw:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '&' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current).strip())

    flags = []
    version_parts = []
    for part in parts:
        if not part:
            continue
        if part in _KNOWN_FLAGS:
            flags.append(part)
        else:
            version_parts.append(part)

    constraint = " & ".join(version_parts) if version_parts else None
    return {"constraint": constraint, "flags": flags}


def _parse_dependency_list(text: str, pos: int) -> tuple:
    """Parse a dependency list starting at '['.

    Dependencies look like:
      [ "ocaml" {>= "4.08"}
        "dune" {>= "2.0"}
        "cppo" {build}
      ]

    Returns (list_of_dep_dicts, new_pos).
    """
    deps = []
    i = pos + 1  # skip '['
    while i < len(text):
        c = text[i]
        if c == ']':
            return deps, i + 1
        elif c == '"':
            # Package name
            name, i = _parse_string(text, i)
            # Look for optional constraint in braces
            constraint = None
            # Skip whitespace
            while i < len(text) and text[i] in ' \t\n\r':
                i += 1
            if i < len(text) and text[i] == '{':
                # Parse constraint
                brace_depth = 1
                j = i + 1
                while j < len(text) and brace_depth > 0:
                    if text[j] == '{':
                        brace_depth += 1
                    elif text[j] == '}':
                        brace_depth -= 1
                    j += 1
                constraint_raw = text[i + 1:j - 1].strip()
                # Clean up: remove quotes around version numbers
                constraint_str = constraint_raw.replace('"', '')
                parsed = _parse_constraint(constraint_str) if constraint_str else {"constraint": None, "flags": []}
                deps.append({"package": name, **parsed})
                i = j
            else:
                deps.append({"package": name, "constraint": None, "flags": []})
        elif c in ' \t\n\r':
            i += 1
        else:
            i += 1
    return deps, i


def _find_matching_bracket(text: str, pos: int) -> int:
    """Find the matching ']' for a '[' at pos, handling nesting.

    Returns the index of the matching ']', or -1 if not found.
    """
    depth = 0
    i = pos
    in_string = False
    while i < len(text):
        c = text[i]
        if in_string:
            if c == '\\' and i + 1 < len(text):
                i += 2
                continue
            if c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _parse_pin_depends_list(text: str, pos: int) -> tuple:
    """Parse a pin-depends list starting at '['.

    pin-depends has nested brackets:
      [ ["pkg.version" "url"]
        ["pkg2.version" "url2"]
      ]

    Returns (list_of_pin_dicts, new_pos).
    """
    pins = []
    i = pos + 1  # skip outer '['
    while i < len(text):
        c = text[i]
        if c == ']':
            return pins, i + 1
        elif c == '[':
            # Inner pair
            inner, i = _parse_list(text, i)
            if len(inner) >= 2:
                name_version = inner[0]
                url = inner[1]
                # Split "name.version" on the last '.'
                dot = name_version.rfind('.')
                if dot > 0:
                    name = name_version[:dot]
                    version = name_version[dot + 1:]
                else:
                    name = name_version
                    version = "dev"
                pins.append({"package": name, "version": version, "url": url})
        elif c in ' \t\n\r':
            i += 1
        else:
            i += 1
    return pins, i


def parse_opam_file(text: str) -> Dict[str, Any]:
    """Parse an opam file into a JSON-friendly dict.

    Returns a dict with keys: synopsis, description, depends, depopts,
    authors, license, homepage.
    """
    result: Dict[str, Any] = {
        "synopsis": "",
        "description": "",
        "depends": [],
        "depopts": [],
        "authors": [],
        "license": "",
        "homepage": "",
        "pin_depends": [],
    }

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # synopsis
        if line.startswith('synopsis:'):
            rest = line[len('synopsis:'):].strip()
            if rest.startswith('"'):
                result["synopsis"], _ = _parse_string(rest, 0)
            i += 1
            continue

        # description
        if line.startswith('description:'):
            rest = line[len('description:'):].strip()
            if rest.startswith('"""'):
                # Multi-line: collect until closing """
                # Rejoin from current position
                full = '\n'.join(lines[i:])
                offset = full.index('"""')
                val, _ = _parse_string(full, offset)
                result["description"] = val.strip()
                # Skip past the closing """
                end_marker = full.find('"""', offset + 3)
                if end_marker != -1:
                    consumed = full[:end_marker + 3].count('\n')
                    i += consumed + 1
                else:
                    i = len(lines)
                continue
            elif rest.startswith('"'):
                result["description"], _ = _parse_string(rest, 0)
            i += 1
            continue

        # depends / depopts
        if line.startswith('depends:') or line.startswith('depopts:'):
            field = "depends" if line.startswith('depends:') else "depopts"
            rest = line[len(field) + 1:].strip()
            if '[' in rest:
                # Might start on this line
                full = '\n'.join(lines[i:])
                bracket_pos = full.index('[')
                deps, _ = _parse_dependency_list(full, bracket_pos)
                result[field] = deps
                # Count lines consumed
                end_bracket = full.find(']', bracket_pos)
                if end_bracket != -1:
                    consumed = full[:end_bracket + 1].count('\n')
                    i += consumed + 1
                else:
                    i = len(lines)
            else:
                # List starts on next line
                i += 1
                if i < len(lines) and '[' in lines[i]:
                    full = '\n'.join(lines[i:])
                    bracket_pos = full.index('[')
                    deps, _ = _parse_dependency_list(full, bracket_pos)
                    result[field] = deps
                    end_bracket = full.find(']', bracket_pos)
                    if end_bracket != -1:
                        consumed = full[:end_bracket + 1].count('\n')
                        i += consumed + 1
                    else:
                        i = len(lines)
                else:
                    i += 1
            continue

        # authors (can be a string or a list)
        if line.startswith('authors:'):
            rest = line[len('authors:'):].strip()
            if rest.startswith('['):
                full = '\n'.join(lines[i:])
                bracket_pos = full.index('[')
                items, _ = _parse_list(full, bracket_pos)
                result["authors"] = items
                end_bracket = full.find(']', bracket_pos)
                if end_bracket != -1:
                    consumed = full[:end_bracket + 1].count('\n')
                    i += consumed + 1
                else:
                    i = len(lines)
            elif rest.startswith('"'):
                author, _ = _parse_string(rest, 0)
                result["authors"] = [author]
            else:
                i += 1
            continue

        # license
        if line.startswith('license:'):
            rest = line[len('license:'):].strip()
            if rest.startswith('"'):
                result["license"], _ = _parse_string(rest, 0)
            i += 1
            continue

        # homepage
        if line.startswith('homepage:'):
            rest = line[len('homepage:'):].strip()
            if rest.startswith('"'):
                result["homepage"], _ = _parse_string(rest, 0)
            i += 1
            continue

        # pin-depends
        if line.startswith('pin-depends:'):
            rest = line[len('pin-depends:'):].strip()
            if '[' in rest:
                full = '\n'.join(lines[i:])
                bracket_pos = full.index('[')
                pins, _ = _parse_pin_depends_list(full, bracket_pos)
                result["pin_depends"] = pins
                end_bracket = _find_matching_bracket(full, bracket_pos)
                if end_bracket != -1:
                    consumed = full[:end_bracket + 1].count('\n')
                    i += consumed + 1
                else:
                    i = len(lines)
            else:
                i += 1
                if i < len(lines) and '[' in lines[i]:
                    full = '\n'.join(lines[i:])
                    bracket_pos = full.index('[')
                    pins, _ = _parse_pin_depends_list(full, bracket_pos)
                    result["pin_depends"] = pins
                    end_bracket = _find_matching_bracket(full, bracket_pos)
                    if end_bracket != -1:
                        consumed = full[:end_bracket + 1].count('\n')
                        i += consumed + 1
                    else:
                        i = len(lines)
                else:
                    i += 1
            continue

        i += 1

    return result
