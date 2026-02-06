#!/usr/bin/env python3
"""
MCP Server for OCaml Documentation

Provides tools for browsing and searching OCaml package documentation
by querying sage.ci.dev (the OCaml docs backend) and Sherlodoc directly.
No local files or embedding server required.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import aiohttp
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP

from version_utils import find_latest_version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAGE_BASE = "https://sage.ci.dev/current/p"

_local_docs_root: Optional[Path] = None

# ---------------------------------------------------------------------------
# Shared HTTP session
# ---------------------------------------------------------------------------

_session: Optional[aiohttp.ClientSession] = None


async def get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


@asynccontextmanager
async def lifespan(server: FastMCP):
    yield
    global _session
    if _session and not _session.closed:
        await _session.close()
        logger.info("HTTP session closed")


mcp = FastMCP("ocaml-docs", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------

_cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)


def cache_get(key: str):
    entry = _cache.get(key)
    if entry and entry[1] > time.time():
        return entry[0]
    return None


def cache_set(key: str, value, ttl: float):
    _cache[key] = (value, time.time() + ttl)


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

async def fetch_text(url: str) -> Optional[str]:
    session = await get_session()
    async with session.get(url) as resp:
        if resp.status != 200:
            return None
        return await resp.text()


async def fetch_json(url: str) -> Optional[Any]:
    session = await get_session()
    async with session.get(url) as resp:
        if resp.status != 200:
            return None
        return await resp.json(content_type=None)


# ---------------------------------------------------------------------------
# Directory listing parser (Apache-style auto-index)
# ---------------------------------------------------------------------------

def parse_directory_listing(html: str) -> List[str]:
    """Extract directory names from an Apache auto-index page."""
    soup = BeautifulSoup(html, "html.parser")
    names = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        # Skip parent directory and non-directory links
        if href.startswith("?") or href.startswith("/") or href == "../":
            continue
        if href.endswith("/"):
            names.append(href.rstrip("/"))
    return names


# ---------------------------------------------------------------------------
# Package list (cached 1 hour)
# ---------------------------------------------------------------------------

async def get_all_packages() -> List[str]:
    cached = cache_get("all_packages")
    if cached is not None:
        return cached
    html = await fetch_text(f"{SAGE_BASE}/")
    if html is None:
        return []
    packages = parse_directory_listing(html)
    cache_set("all_packages", packages, 3600)
    return packages


# ---------------------------------------------------------------------------
# Version resolution (cached 30 min)
# ---------------------------------------------------------------------------

async def resolve_version(package: str, version: Optional[str] = None) -> Optional[str]:
    if version:
        return version
    cache_key = f"versions:{package}"
    versions = cache_get(cache_key)
    if versions is None:
        html = await fetch_text(f"{SAGE_BASE}/{package}/")
        if html is None:
            return None
        versions = parse_directory_listing(html)
        cache_set(cache_key, versions, 1800)
    if not versions:
        return None
    latest, _ = find_latest_version(versions)
    return latest


# ---------------------------------------------------------------------------
# status.json (cached 30 min)
# ---------------------------------------------------------------------------

async def get_status(package: str, version: str) -> Optional[Dict]:
    cache_key = f"status:{package}/{version}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    data = await fetch_json(f"{SAGE_BASE}/{package}/{version}/status.json")
    if data is not None:
        cache_set(cache_key, data, 1800)
    return data


# ---------------------------------------------------------------------------
# Doc JSON (cached 10 min)
# ---------------------------------------------------------------------------

async def get_doc_json(package: str, version: str, path: str) -> Optional[Dict]:
    cache_key = f"doc:{package}/{version}/{path}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    data = await fetch_json(f"{SAGE_BASE}/{package}/{version}/{path}")
    if data is not None:
        cache_set(cache_key, data, 600)
    return data


# ---------------------------------------------------------------------------
# HTML-to-text extraction for odoc content
# ---------------------------------------------------------------------------

def extract_preamble_text(html: str) -> str:
    """Extract plain text from an odoc preamble HTML fragment."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    parts = []
    for p in soup.find_all("p"):
        parts.append(p.get_text(strip=True))
    return " ".join(parts).strip()


def extract_specs(html: str, limit: int = 100) -> tuple:
    """Extract spec items (values, types, modules, etc.) from odoc content HTML.

    Returns (items, truncated) where truncated is True if limit was hit.
    """
    if not html:
        return [], False
    soup = BeautifulSoup(html, "html.parser")
    items = []
    truncated = False

    for spec_div in soup.find_all("div", class_="spec"):
        if len(items) >= limit:
            truncated = True
            break

        anchor = spec_div.get("id", "")
        code = spec_div.find("code")
        signature = code.get_text(strip=True) if code else ""

        # Get doc from next sibling
        doc = ""
        doc_div = spec_div.find_next_sibling("div", class_="spec-doc")
        if doc_div:
            doc_parts = []
            for p in doc_div.find_all("p"):
                doc_parts.append(p.get_text(strip=True))
            doc = " ".join(doc_parts).strip()

        # Categorize by anchor prefix
        kind = "other"
        name = anchor
        if anchor.startswith("val-"):
            kind, name = "val", anchor[4:]
        elif anchor.startswith("type-"):
            kind, name = "type", anchor[5:]
        elif anchor.startswith("module-type-"):
            kind, name = "module type", anchor[12:]
        elif anchor.startswith("module-"):
            kind, name = "module", anchor[7:]
        elif anchor.startswith("exception-"):
            kind, name = "exception", anchor[10:]
        elif anchor.startswith("class-"):
            kind, name = "class", anchor[6:]

        items.append({"kind": kind, "name": name, "signature": signature, "doc": doc})

    return items, truncated


def extract_package_libraries(html: str) -> List[Dict[str, Any]]:
    """Extract library and module listings from a package doc page.

    The package index page lists libraries as h2 headings, each followed by
    a list of modules. Some simpler packages just have a flat module list.
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    libraries = []
    current_lib = None

    for element in soup.children:
        if not hasattr(element, "name") or element.name is None:
            continue

        # A h2 typically starts a library section
        if element.name == "h2":
            text = element.get_text(strip=True)
            # Library headings usually contain "Library <name>"
            if current_lib:
                libraries.append(current_lib)
            current_lib = {"name": text, "modules": []}

        # Module listings come in <ul> or <dl> after the heading
        if element.name in ("ul", "dl"):
            modules = []
            for li in element.find_all("li"):
                link = li.find("a")
                if link:
                    mod_name = link.get_text(strip=True)
                    # Synopsis is any text after the link
                    full_text = li.get_text(strip=True)
                    synopsis = full_text[len(mod_name):].strip().lstrip(":").strip()
                    modules.append({"name": mod_name, "synopsis": synopsis})
            if modules:
                if current_lib is None:
                    current_lib = {"name": "default", "modules": []}
                current_lib["modules"].extend(modules)

    if current_lib:
        libraries.append(current_lib)

    return libraries


# ---------------------------------------------------------------------------
# Tool 1: sherlodoc
# ---------------------------------------------------------------------------

@mcp.tool()
async def sherlodoc(query: str) -> Dict[str, Any]:
    """Search OCaml names and type signatures across all packages using Sherlodoc.

    Good for finding functions by type signature or name.

    Args:
        query: A type signature like "int -> string", a name like "List.map",
               or a type like "'a list -> ('a -> 'b) -> 'b list"

    Returns:
        Matching entries with signatures and documentation
    """
    try:
        encoded = quote(query)
        url = f"https://doc.sherlocode.com/api?q={encoded}"
        session = await get_session()
        async with session.get(url) as resp:
            if resp.status != 200:
                return {"error": f"Sherlodoc returned status {resp.status}"}
            html = await resp.text()

        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.find_all("li")[:20]:
            result = {}
            pre = item.find("pre")
            if pre:
                result["signature"] = pre.get_text(strip=True)
                link = pre.find("a")
                if link and link.get("href"):
                    result["url"] = link["href"]
                    em = link.find("em")
                    if em:
                        result["module_path"] = em.get_text()

            comment = item.find("div", class_="comment")
            if comment:
                doc_parts = [p.get_text(strip=True) for p in comment.find_all("p")]
                if doc_parts:
                    result["documentation"] = " ".join(doc_parts)

            if result.get("signature"):
                results.append(result)

        return {"query": query, "results": results, "total_results": len(results)}

    except aiohttp.ClientError as e:
        return {"error": f"Failed to connect to Sherlodoc: {e}"}
    except Exception as e:
        return {"error": f"Sherlodoc search failed: {e}"}


# ---------------------------------------------------------------------------
# Tool 2: search_package_names
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_package_names(query: str) -> Dict[str, Any]:
    """Find OCaml packages by name.

    Searches the full package list on sage.ci.dev for case-insensitive
    substring matches.

    Args:
        query: Substring to search for in package names, e.g. "lwt", "http", "json"

    Returns:
        List of matching package names (up to 50)
    """
    try:
        packages = await get_all_packages()
        q = query.lower()
        matches = [p for p in packages if q in p.lower()]
        return {
            "query": query,
            "matches": matches[:50],
            "total_matches": len(matches),
        }
    except Exception as e:
        return {"error": f"Package search failed: {e}"}


# ---------------------------------------------------------------------------
# Tool 3: get_package_info
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_package_info(package_name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Get an overview of an OCaml package: description, libraries, and modules.

    Fetches the package's documentation page from sage.ci.dev and extracts
    the README/preamble text plus the list of libraries and their modules.

    Args:
        package_name: Package name, e.g. "lwt", "base", "cohttp"
        version: Optional specific version. Defaults to latest.

    Returns:
        Package name, version, build status, description, and library/module listing
    """
    try:
        ver = await resolve_version(package_name, version)
        if ver is None:
            return {"error": f"Package '{package_name}' not found on sage.ci.dev"}

        status = await get_status(package_name, ver)
        if status is None:
            return {"error": f"Could not fetch status for {package_name}/{ver}"}

        failed = status.get("failed", True)

        # Fetch package doc page
        doc = await get_doc_json(package_name, ver, "doc/index.html.json")

        description = ""
        libraries = []
        if doc:
            preamble = doc.get("preamble", "")
            content = doc.get("content", "")
            # Description comes from preamble first, or first paragraphs of content
            description = extract_preamble_text(preamble)
            if not description:
                description = extract_preamble_text(content)
            libraries = extract_package_libraries(content)

        return {
            "package": package_name,
            "version": ver,
            "failed": failed,
            "description": description,
            "libraries": libraries,
        }
    except Exception as e:
        return {"error": f"Failed to get package info: {e}"}


# ---------------------------------------------------------------------------
# Tool 4: get_module_doc
# ---------------------------------------------------------------------------

def find_module_file(files: List[str], module_path: str) -> Optional[str]:
    """Find the doc file for a module path like 'Base.List' in the files list.

    Tries several matching strategies:
    1. Exact suffix match: Base/List/index.html
    2. Case-insensitive match
    """
    # Convert dot path to directory path
    parts = module_path.split(".")
    suffix = "/".join(parts) + "/index.html"

    # Try exact match
    for f in files:
        if f.endswith(suffix):
            return f

    # Try case-insensitive
    suffix_lower = suffix.lower()
    for f in files:
        if f.lower().endswith(suffix_lower):
            return f

    return None


@mcp.tool()
async def get_module_doc(
    package_name: str, module_path: str, version: Optional[str] = None
) -> Dict[str, Any]:
    """Get documentation for a specific OCaml module.

    Fetches and parses the module's documentation page from sage.ci.dev.
    Returns the preamble, type definitions, values/functions, and submodules
    as structured text.

    Args:
        package_name: Package name, e.g. "lwt", "base"
        module_path: Dot-separated module path, e.g. "Lwt", "Base.List", "Lwt_unix.LargeFile"
        version: Optional specific version. Defaults to latest.

    Returns:
        Module documentation with preamble, types, values, and submodules
    """
    try:
        ver = await resolve_version(package_name, version)
        if ver is None:
            return {"error": f"Package '{package_name}' not found on sage.ci.dev"}

        status = await get_status(package_name, ver)
        if status is None:
            return {"error": f"Could not fetch status for {package_name}/{ver}"}

        files = status.get("files", [])
        matched_file = find_module_file(files, module_path)
        if matched_file is None:
            # List available modules to help the user
            top_modules = set()
            for f in files:
                if f.startswith("doc/") and f.endswith("/index.html"):
                    parts = f[4:].split("/")  # strip "doc/"
                    # parts: [library, Module, ..., "index.html"]
                    # Show the immediate module name (second element)
                    if len(parts) >= 3:
                        top_modules.add(parts[1])
            hint = sorted(top_modules)[:30]
            return {
                "error": f"Module '{module_path}' not found in {package_name}/{ver}",
                "available_top_level": hint,
            }

        # Fetch the JSON doc — the file list has .html paths, we need .html.json
        json_path = matched_file + ".json"
        doc = await get_doc_json(package_name, ver, json_path)
        if doc is None:
            return {"error": f"Could not fetch documentation for {module_path}"}

        preamble = extract_preamble_text(doc.get("preamble", ""))
        content_html = doc.get("content", "")
        specs, truncated = extract_specs(content_html, limit=100)

        result: Dict[str, Any] = {
            "package": package_name,
            "version": ver,
            "module": module_path,
            "preamble": preamble,
            "items": specs,
        }
        if truncated:
            result["truncated"] = True
            result["note"] = "Output truncated at 100 items. The module has more entries."

        return result

    except Exception as e:
        return {"error": f"Failed to get module doc: {e}"}


# ---------------------------------------------------------------------------
# Local odoc tools
# ---------------------------------------------------------------------------

_SKIP_DIRS = {"odoc.support"}


def _scan_local_modules(root: Path) -> List[Dict[str, str]]:
    """Walk the local docs directory and return {library, module_path} entries."""
    results = []
    for lib_dir in sorted(root.iterdir()):
        if not lib_dir.is_dir() or lib_dir.name in _SKIP_DIRS:
            continue
        library = lib_dir.name
        for json_file in sorted(lib_dir.rglob("index.html.json")):
            rel = json_file.relative_to(lib_dir)
            # rel looks like Module/Sub/index.html.json or just index.html.json
            parts = list(rel.parts[:-1])  # drop "index.html.json"
            if not parts:
                # library-level page, not a module
                continue
            module_path = ".".join(parts)
            results.append({"library": library, "module_path": module_path})
    return results


@mcp.tool()
async def list_local_modules() -> Dict[str, Any]:
    """List all modules available in the local odoc documentation.

    Walks the local docs directory (set via --local-docs) and returns
    every module grouped by library.

    Returns:
        List of {library, module_path} entries
    """
    if _local_docs_root is None:
        return {"error": "Local docs not configured. Start the server with --local-docs <path>."}

    cached = cache_get("local_modules")
    if cached is not None:
        return cached

    if not _local_docs_root.is_dir():
        return {"error": f"Local docs path does not exist: {_local_docs_root}"}

    modules = _scan_local_modules(_local_docs_root)
    result = {"modules": modules, "total": len(modules)}
    cache_set("local_modules", result, 300)  # 5 min TTL
    return result


@mcp.tool()
async def get_local_module_doc(module_path: str) -> Dict[str, Any]:
    """Get documentation for a module from the local odoc output.

    Looks up a dot-separated module path (e.g. "Irmin.Store") in the local
    docs directory and returns its preamble and spec items.

    Args:
        module_path: Dot-separated module path, e.g. "Irmin", "Irmin.Store"

    Returns:
        Library name, module path, preamble, and items
    """
    if _local_docs_root is None:
        return {"error": "Local docs not configured. Start the server with --local-docs <path>."}

    if not _local_docs_root.is_dir():
        return {"error": f"Local docs path does not exist: {_local_docs_root}"}

    parts = module_path.split(".")
    suffix = Path(*parts) / "index.html.json"

    # Search across library directories
    for lib_dir in sorted(_local_docs_root.iterdir()):
        if not lib_dir.is_dir() or lib_dir.name in _SKIP_DIRS:
            continue
        candidate = lib_dir / suffix
        if candidate.is_file():
            doc = json.loads(candidate.read_text())
            preamble = extract_preamble_text(doc.get("preamble", ""))
            specs, truncated = extract_specs(doc.get("content", ""), limit=100)
            result: Dict[str, Any] = {
                "library": lib_dir.name,
                "module": module_path,
                "preamble": preamble,
                "items": specs,
            }
            if truncated:
                result["truncated"] = True
                result["note"] = "Output truncated at 100 items. The module has more entries."
            return result

    return {"error": f"Module '{module_path}' not found in local docs"}


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

def main():
    import sys
    import asyncio

    global _local_docs_root

    # Parse --local-docs flag from anywhere in argv
    args = sys.argv[1:]
    if "--local-docs" in args:
        idx = args.index("--local-docs")
        if idx + 1 < len(args):
            _local_docs_root = Path(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --local-docs requires a path argument", file=sys.stderr)
            sys.exit(1)

    if args and args[0] == "--test":
        test_args = args[1:]

        async def run_test():
            if not test_args:
                print("Usage: mcp_server.py [--local-docs <path>] --test <command> [args...]")
                print("Commands:")
                print("  sherlodoc <query>")
                print("  search-packages <query>")
                print("  package-info <package> [version]")
                print("  module-doc <package> <module_path> [version]")
                print("  list-local")
                print("  local-module-doc <module_path>")
                return

            cmd = test_args[0]

            if cmd == "sherlodoc":
                query = test_args[1] if len(test_args) > 1 else "int -> string"
                result = await sherlodoc(query)
            elif cmd == "search-packages":
                query = test_args[1] if len(test_args) > 1 else "http"
                result = await search_package_names(query)
            elif cmd == "package-info":
                pkg = test_args[1] if len(test_args) > 1 else "lwt"
                ver = test_args[2] if len(test_args) > 2 else None
                result = await get_package_info(pkg, ver)
            elif cmd == "module-doc":
                pkg = test_args[1] if len(test_args) > 1 else "lwt"
                mod = test_args[2] if len(test_args) > 2 else "Lwt"
                ver = test_args[3] if len(test_args) > 3 else None
                result = await get_module_doc(pkg, mod, ver)
            elif cmd == "list-local":
                result = await list_local_modules()
            elif cmd == "local-module-doc":
                mod = test_args[1] if len(test_args) > 1 else "Stdlib"
                result = await get_local_module_doc(mod)
            else:
                result = {"error": f"Unknown command: {cmd}"}

            print(json.dumps(result, indent=2))

            # Clean up session
            global _session
            if _session and not _session.closed:
                await _session.close()

        asyncio.run(run_test())
    else:
        mcp.run(transport="sse", host="0.0.0.0", port=8007)


if __name__ == "__main__":
    main()
