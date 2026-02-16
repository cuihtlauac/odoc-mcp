#!/usr/bin/env python3
"""
MCP server for OCaml package documentation.

Queries sage.ci.dev and Sherlodoc for any published package.
Can also browse locally-built odoc output.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import time
from contextlib import asynccontextmanager
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import quote

import aiohttp
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import FunctionResource

from opam_parser import parse_opam_file
from sexp_parser import find_stanzas, parse_sexp
from version_utils import compare_versions, find_latest_version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAGE_BASE = "https://sage.ci.dev/current/p"

OPAM_REPO_OWNER = "ocaml"
OPAM_REPO_NAME = "opam-repository"
_local_docs_root: Optional[Path] = None


def _github_headers() -> Dict[str, str]:
    """Return GitHub auth headers if GITHUB_TOKEN is set."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return {"Authorization": f"token {token}"}
    return {}

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
# Documentation source protocol and registry
# ---------------------------------------------------------------------------

class DocSource(Protocol):
    name: str
    description: str
    priority: int  # lower = tried first

    async def get_module_doc(self, module_path: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Return doc dict if found, None if this source doesn't have it."""
        ...


_doc_sources: Dict[str, "DocSource"] = {}


def register_doc_source(name: str, handler: "DocSource"):
    """Register a documentation source and expose it as an MCP resource."""
    _doc_sources[name] = handler

    async def read_meta() -> str:
        return json.dumps({
            "name": handler.name,
            "description": handler.description,
            "priority": handler.priority,
        })

    resource = FunctionResource.from_function(
        fn=read_meta,
        uri=f"ocaml-docs://{name}",
        name=f"ocaml-docs-{name}",
        description=handler.description,
        mime_type="application/json",
    )
    mcp.add_resource(resource)


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

async def fetch_text(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[str]:
    session = await get_session()
    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            return None
        return await resp.text()


async def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Optional[Any]:
    session = await get_session()
    async with session.get(url, headers=headers) as resp:
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
# Tool: ocaml_search
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_search")
async def sherlodoc(query: str) -> Dict[str, Any]:
    """Search OCaml names and type signatures across all packages.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

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
# Tool: ocaml_package_search
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_package_search")
async def search_package_names(
    query: str, repos: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Find OCaml packages by name.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Searches both sage.ci.dev (documentation site) and opam repositories
    (GitHub) for case-insensitive substring matches, returning consolidated
    results with their sources.

    Args:
        query: Substring to search for in OCaml package names, e.g. "lwt", "http", "json"
        repos: Optional list of opam GitHub repository URLs to search.
               Defaults to the main opam-repository.

    Returns:
        Matching packages (up to 50) with their sources
    """
    if repos is None:
        repos = [OPAM_REPO_URL]

    try:
        q = query.lower()
        merged: Dict[str, List[str]] = {}

        # Search sage.ci.dev
        sage_packages = await get_all_packages()
        for p in sage_packages:
            if q in p.lower():
                merged.setdefault(p, []).append("sage")

        # Search opam repos
        for repo_url in repos:
            if _github_api_url(repo_url) is None:
                return {"error": f"Not a valid GitHub repository URL: {repo_url!r}"}
            opam_packages = await get_opam_all_packages(repo_url)
            for p in opam_packages:
                if q in p.lower():
                    merged.setdefault(p, []).append(repo_url)

        matches = [
            {"package": name, "sources": sources}
            for name, sources in sorted(merged.items())
        ]
        return {
            "query": query,
            "matches": matches[:50],
            "total_matches": len(merged),
        }
    except Exception as e:
        return {"error": f"Package search failed: {e}"}


# ---------------------------------------------------------------------------
# Tool: ocaml_package_doc
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_package_doc")
async def get_package_info(package_name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Get an overview of an OCaml package: description, libraries, and modules.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Fetches the package's documentation page from sage.ci.dev and extracts
    the README/preamble text plus the list of libraries and their modules.

    Args:
        package_name: OCaml package name, e.g. "lwt", "base", "cohttp"
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
# Module doc helpers
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


# ---------------------------------------------------------------------------
# SageDocSource
# ---------------------------------------------------------------------------

class SageDocSource:
    name = "sage"
    description = "OCaml module docs from sage.ci.dev"
    priority = 10

    async def get_module_doc(self, module_path: str, **kwargs) -> Optional[Dict[str, Any]]:
        package_name = kwargs.get("package_name")
        version = kwargs.get("version")
        if not package_name:
            return None
        try:
            ver = await resolve_version(package_name, version)
            if ver is None:
                return None
            status = await get_status(package_name, ver)
            if status is None:
                return None
            files = status.get("files", [])
            matched_file = find_module_file(files, module_path)
            if matched_file is None:
                return None
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
                result["note"] = "Output truncated at 100 items."
            return result
        except Exception as e:
            return {"error": f"Failed to get module doc from sage: {e}"}


register_doc_source("sage", SageDocSource())


# ---------------------------------------------------------------------------
# Local odoc helpers
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


# ---------------------------------------------------------------------------
# Tool: ocaml_module_list_local
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_module_list_local")
async def list_local_modules() -> Dict[str, Any]:
    """List all modules available in the local OCaml odoc documentation.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

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


# ---------------------------------------------------------------------------
# LocalDocSource
# ---------------------------------------------------------------------------

class LocalDocSource:
    name = "local"
    description = "OCaml module docs from local odoc output"
    priority = 0

    def __init__(self, root: Path):
        self.root = root

    async def get_module_doc(self, module_path: str, **kwargs) -> Optional[Dict[str, Any]]:
        if not self.root.is_dir():
            return None
        parts = module_path.split(".")
        suffix = Path(*parts) / "index.html.json"
        for lib_dir in sorted(self.root.iterdir()):
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
                    result["note"] = "Output truncated at 100 items."
                return result
        return None


# ---------------------------------------------------------------------------
# Tool: ocaml_module_doc
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_module_doc")
async def get_module_doc(
    module_path: str,
    package_name: Optional[str] = None,
    version: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Get documentation for a specific OCaml module.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Fetches and parses the module's documentation page from sage.ci.dev.
    Returns the preamble, type definitions, values/functions, and submodules
    as structured text.

    Args:
        module_path: Dot-separated OCaml module path, e.g. "Lwt", "Base.List", "Lwt_unix.LargeFile"
        package_name: OCaml package name, e.g. "lwt", "base". Required for sage.ci.dev lookups.
        version: Optional specific version. Defaults to latest.
        source: Optional source name (e.g. "sage", "local"). If omitted, tries all sources in priority order.

    Returns:
        Module documentation with preamble, types, values, and submodules
    """
    if source is not None:
        handler = _doc_sources.get(source)
        if handler is None:
            available = sorted(_doc_sources.keys())
            return {"error": f"Unknown source: {source!r}. Available: {available}"}
        result = await handler.get_module_doc(
            module_path, package_name=package_name, version=version
        )
        if result is None:
            return {"error": f"Module '{module_path}' not found in source '{source}'"}
        result["source"] = handler.name
        return result

    # Try all sources in priority order (lowest number first)
    for handler in sorted(_doc_sources.values(), key=lambda s: s.priority):
        result = await handler.get_module_doc(
            module_path, package_name=package_name, version=version
        )
        if result is not None:
            if "error" in result:
                return result
            result["source"] = handler.name
            return result

    return {"error": f"Module '{module_path}' not found in any source"}


# ---------------------------------------------------------------------------
# opam repository helpers (GitHub)
# ---------------------------------------------------------------------------

def _parse_github_url(repo_url: str) -> Optional[tuple]:
    """Parse owner and repo name from a GitHub URL.

    Accepts URLs like https://github.com/owner/repo or
    https://github.com/owner/repo.git. Returns (owner, repo) or None.
    """
    m = re.match(r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', repo_url)
    if not m:
        return None
    return m.group(1), m.group(2)


def _github_api_url(repo_url: str) -> Optional[str]:
    """Return the GitHub API base URL for a repository URL, or None."""
    parsed = _parse_github_url(repo_url)
    if not parsed:
        return None
    return f"https://api.github.com/repos/{parsed[0]}/{parsed[1]}"


def _github_raw_url(repo_url: str) -> Optional[str]:
    """Return the raw.githubusercontent.com base URL for a repository, or None."""
    parsed = _parse_github_url(repo_url)
    if not parsed:
        return None
    return f"https://raw.githubusercontent.com/{parsed[0]}/{parsed[1]}"


async def get_opam_all_packages(repo_url: str) -> List[str]:
    """Fetch all package names from an opam repository via GitHub Trees API.

    Cached for 1 hour per repo URL.
    """
    cache_key = f"opam_all_packages:{repo_url}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    api_base = _github_api_url(repo_url)
    if api_base is None:
        return []

    url = f"{api_base}/git/trees/master?recursive=0"
    data = await fetch_json(url, headers=_github_headers())
    if data is None:
        return []

    # Find the "packages" tree sha
    packages_sha = None
    for item in data.get("tree", []):
        if item.get("path") == "packages" and item.get("type") == "tree":
            packages_sha = item["sha"]
            break

    if packages_sha is None:
        return []

    # Fetch the packages tree (just one level deep)
    pkg_url = f"{api_base}/git/trees/{packages_sha}"
    pkg_data = await fetch_json(pkg_url, headers=_github_headers())
    if pkg_data is None:
        return []

    names = sorted(
        item["path"]
        for item in pkg_data.get("tree", [])
        if item.get("type") == "tree"
    )
    cache_set(cache_key, names, 3600)
    return names


async def opam_resolve_version(
    repo_url: str, package: str, version: Optional[str] = None
) -> Optional[str]:
    """Resolve a version for a package from an opam repository on GitHub.

    If version is given, return it as-is.
    Otherwise, list all versions via the GitHub Contents API and pick the latest.
    """
    if version:
        return version

    api_base = _github_api_url(repo_url)
    if api_base is None:
        return None

    url = f"{api_base}/contents/packages/{package}"
    data = await fetch_json(url, headers=_github_headers())
    if data is None or not isinstance(data, list):
        return None

    prefix = f"{package}."
    versions = [
        item["name"][len(prefix):]
        for item in data
        if item.get("type") == "dir" and item["name"].startswith(prefix)
    ]
    if not versions:
        return None

    latest, _ = find_latest_version(versions)
    return latest


OPAM_REPO_URL = f"https://github.com/{OPAM_REPO_OWNER}/{OPAM_REPO_NAME}"


# ---------------------------------------------------------------------------
# Tool: ocaml_package_versions
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_package_versions")
async def opam_list_versions(package_name: str, repo: str) -> Dict[str, Any]:
    """List all versions of an OCaml opam package, newest first.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Fetches version directories from the given opam repository on GitHub.

    Args:
        package_name: OCaml package name, e.g. "lwt", "dune"
        repo: GitHub repository URL, e.g. "https://github.com/ocaml/opam-repository"

    Returns:
        List of versions sorted newest-first
    """
    try:
        api_base = _github_api_url(repo)
        if api_base is None:
            return {"error": f"Not a valid GitHub repository URL: {repo!r}"}

        url = f"{api_base}/contents/packages/{package_name}"
        data = await fetch_json(url, headers=_github_headers())
        if data is None:
            return {"error": f"Package '{package_name}' not found in {repo}"}
        if not isinstance(data, list):
            return {"error": f"Unexpected response for '{package_name}'"}

        prefix = f"{package_name}."
        versions = [
            item["name"][len(prefix):]
            for item in data
            if item.get("type") == "dir" and item["name"].startswith(prefix)
        ]
        versions.sort(key=cmp_to_key(compare_versions), reverse=True)
        return {
            "package": package_name,
            "versions": versions,
            "total": len(versions),
        }
    except Exception as e:
        return {"error": f"Failed to list versions: {e}"}


# ---------------------------------------------------------------------------
# Tool: ocaml_package_meta
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_package_meta")
async def opam_show_package(
    package_name: str, repo: str, version: Optional[str] = None
) -> Dict[str, Any]:
    """Show details of an OCaml opam package by parsing its opam file.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Fetches the opam file from the given opam repository on GitHub. If no
    version is given, uses the latest.

    Args:
        package_name: OCaml package name, e.g. "lwt"
        repo: GitHub repository URL, e.g. "https://github.com/ocaml/opam-repository"
        version: Optional version string. Defaults to latest.

    Returns:
        Synopsis, description, dependencies with constraints, optional deps,
        authors, license, homepage
    """
    try:
        raw_base = _github_raw_url(repo)
        if raw_base is None:
            return {"error": f"Not a valid GitHub repository URL: {repo!r}"}

        ver = await opam_resolve_version(repo, package_name, version)
        if ver is None:
            return {"error": f"Package '{package_name}' not found in {repo}"}

        url = (
            f"{raw_base}/master/packages/"
            f"{package_name}/{package_name}.{ver}/opam"
        )
        text = await fetch_text(url, headers=_github_headers())
        if text is None:
            return {
                "error": f"Could not fetch opam file for {package_name}.{ver}"
            }

        parsed = parse_opam_file(text)
        return {"package": package_name, "version": ver, **parsed}

    except Exception as e:
        return {"error": f"Failed to show package: {e}"}


# ---------------------------------------------------------------------------
# opam CLI helper
# ---------------------------------------------------------------------------

async def _run_opam(*args: str, timeout: float = 15.0) -> Optional[str]:
    """Run an opam command and return its stdout, or None on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "opam", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode != 0:
            return None
        return stdout.decode().strip()
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return None


# ---------------------------------------------------------------------------
# Tool: ocaml_deps_managers
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_deps_managers")
async def detect_dependency_managers() -> Dict[str, Any]:
    """Detect which OCaml dependency managers (opam, dune-pkg) are active for the current project.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Checks for dune-pkg and opam setups. Returns a list of detected managers
    with details about each.

    Returns:
        List of detected managers ("opam", "dune-pkg") with details
    """
    cwd = Path.cwd()
    managers = []

    # --- dune-pkg detection ---
    dune_pkg_info: Dict[str, Any] = {"manager": "dune-pkg"}
    dune_pkg_detected = False

    lock_dir = cwd / "dune.lock"
    if lock_dir.is_dir():
        dune_pkg_detected = True
        dune_pkg_info["lock_dir"] = str(lock_dir)

    workspace = cwd / "dune-workspace"
    if workspace.is_file():
        try:
            ws_text = workspace.read_text()
            if "(pkg" in ws_text:
                dune_pkg_detected = True
                dune_pkg_info["workspace_pkg_enabled"] = True
        except OSError:
            pass

    if dune_pkg_detected:
        managers.append(dune_pkg_info)

    # --- opam detection ---
    opam_info: Dict[str, Any] = {"manager": "opam"}
    opam_detected = False

    # Local switch
    local_opam = cwd / "_opam"
    if local_opam.is_dir():
        opam_detected = True
        opam_info["local_switch"] = str(local_opam)

    # Global opam root
    opam_root = os.environ.get("OPAMROOT", os.path.expanduser("~/.opam"))
    if Path(opam_root).is_dir():
        opam_detected = True
        opam_info["opam_root"] = opam_root

    # OPAMSWITCH env var
    opam_switch_env = os.environ.get("OPAMSWITCH")
    if opam_switch_env:
        opam_detected = True
        opam_info["opam_switch_env"] = opam_switch_env

    # System OCaml (outside opam)
    if not opam_detected:
        for tool in ("ocamlfind", "ocamlc"):
            if shutil.which(tool):
                opam_detected = True
                opam_info["system_ocaml"] = tool
                break

    if opam_detected:
        managers.append(opam_info)

    result: Dict[str, Any] = {
        "managers": [m["manager"] for m in managers],
        "details": managers,
    }
    if len(managers) > 1:
        result["warning"] = (
            "Both opam and dune-pkg detected. Having both active risks "
            "inconsistent dependency sets."
        )
    return result


# ---------------------------------------------------------------------------
# Tool: ocaml_deps_status
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_deps_status")
async def dependency_environment_status() -> Dict[str, Any]:
    """Report the current OCaml dependency environment status.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Calls detect_dependency_managers() and adds details: opam switch info,
    env consistency, dune lock status.

    Returns:
        Dependency manager details, switch info, and any warnings
    """
    dm = await detect_dependency_managers()
    result: Dict[str, Any] = dict(dm)
    active = dm.get("managers", [])

    if "opam" in active:
        opam_details: Dict[str, Any] = {}

        # Current switch
        switch = await _run_opam("var", "switch")
        if switch:
            opam_details["current_switch"] = switch

        # Switch kind
        cwd = Path.cwd()
        if (cwd / "_opam").is_dir():
            opam_details["switch_kind"] = "local"
        else:
            opam_details["switch_kind"] = "global"

        # Env consistency
        env_prefix = os.environ.get("OPAM_SWITCH_PREFIX")
        if switch and env_prefix:
            opam_var_prefix = await _run_opam("var", "prefix")
            if opam_var_prefix and env_prefix != opam_var_prefix:
                opam_details["env_mismatch"] = {
                    "OPAM_SWITCH_PREFIX": env_prefix,
                    "opam_var_prefix": opam_var_prefix,
                    "fix": "Run: eval $(opam env)",
                }

        # Available switches
        switches_raw = await _run_opam("switch", "list", "--short")
        if switches_raw:
            opam_details["available_switches"] = switches_raw.splitlines()

        result["opam"] = opam_details

    if "dune-pkg" in active:
        dune_details: Dict[str, Any] = {}
        lock_dir = Path.cwd() / "dune.lock"
        if lock_dir.is_dir():
            dune_details["locked"] = True
        else:
            dune_details["locked"] = False
            dune_details["note"] = "Run `dune pkg lock` to create a lock directory."
        result["dune_pkg"] = dune_details

    if not active:
        result["note"] = "No dependency manager detected for this project."

    return result


# ---------------------------------------------------------------------------
# Tool: ocaml_deps_installed
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_deps_installed")
async def list_installed_packages(
    source: str, switch: Optional[str] = None
) -> Dict[str, Any]:
    """List OCaml packages installed in the current project (via opam or dune-pkg).

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Args:
        source: One of "dune-pkg", "opam-switch", or "opam-system".
        switch: Optional opam switch name (only used with "opam-switch").

    Returns:
        List of installed OCaml packages with names and versions
    """
    if source == "dune-pkg":
        lock_dir = Path.cwd() / "dune.lock"
        if not lock_dir.is_dir():
            return {"error": "No dune.lock/ directory found in the current project."}

        packages = []
        for pkg_file in sorted(lock_dir.iterdir()):
            if pkg_file.suffix == ".pkg":
                name = pkg_file.stem
                # Try to extract version from the lock file
                version = None
                try:
                    content = pkg_file.read_text()
                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith("(version"):
                            # (version 1.2.3)
                            version = line.split()[1].rstrip(")")
                            break
                except OSError:
                    pass
                packages.append({"name": name, "version": version})

        return {"source": "dune-pkg", "packages": packages, "total": len(packages)}

    elif source == "opam-switch":
        args = ["list", "--installed",
                "--columns=name,version,synopsis", "--separator=\t"]
        if switch:
            args = ["--switch", switch] + args
        raw = await _run_opam(*args)
        if raw is None:
            return {"error": "Failed to run opam list. Is opam installed?"}

        packages = []
        for line in raw.splitlines():
            # Skip header lines (contain "# " prefix or are dashes)
            if line.startswith("#") or set(line.strip()) <= {"-", " ", "\t"}:
                continue
            parts = line.split("\t", 2)
            if len(parts) >= 2:
                pkg: Dict[str, Any] = {"name": parts[0].strip(), "version": parts[1].strip()}
                if len(parts) >= 3:
                    pkg["synopsis"] = parts[2].strip()
                packages.append(pkg)

        return {
            "source": "opam-switch",
            "switch": switch,
            "packages": packages,
            "total": len(packages),
        }

    elif source == "opam-system":
        ocamlfind = shutil.which("ocamlfind")
        if not ocamlfind:
            return {"error": "ocamlfind not found on PATH."}

        try:
            proc = await asyncio.create_subprocess_exec(
                ocamlfind, "list",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15.0)
            raw = stdout.decode().strip()
        except (asyncio.TimeoutError, OSError) as e:
            return {"error": f"Failed to run ocamlfind list: {e}"}

        packages = []
        for line in raw.splitlines():
            # Format: "package_name  (version: x.y.z)"
            match = re.match(r'^(\S+)\s+\(version:\s*([^)]*)\)', line)
            if match:
                packages.append({"name": match.group(1), "version": match.group(2)})
            elif line.strip():
                packages.append({"name": line.strip().split()[0], "version": None})

        return {"source": "opam-system", "packages": packages, "total": len(packages)}

    else:
        return {"error": f"Unknown source: {source!r}. Use 'dune-pkg', 'opam-switch', or 'opam-system'."}


# ---------------------------------------------------------------------------
# Tool: ocaml_deps_pins
# ---------------------------------------------------------------------------

def _parse_opam_pin_list(raw: str) -> List[Dict[str, Any]]:
    """Parse output of 'opam pin list'.

    Format: name.version  [(uninstalled)]  kind  url  [(at hash)]
    """
    pins = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if not parts:
            continue
        name_version = parts[0]
        # Package names don't contain dots, so split on the first one
        dot = name_version.find(".")
        if dot > 0:
            name = name_version[:dot]
            version = name_version[dot + 1:]
        else:
            name = name_version
            version = "dev"
        # Skip flags like (uninstalled) and (at hash) to find kind and url
        rest = [p for p in parts[1:] if not p.startswith("(") and not p.endswith(")")]
        kind = rest[0] if len(rest) > 0 else None
        url = rest[1] if len(rest) > 1 else None
        pin: Dict[str, Any] = {
            "package": name,
            "version": version,
            "origin": "opam-switch",
        }
        if kind:
            pin["kind"] = kind
        if url:
            pin["url"] = url
        pins.append(pin)
    return pins


def _pins_from_opam_files(cwd: Path) -> List[Dict[str, Any]]:
    """Extract pin-depends from .opam files in cwd."""
    pins = []
    for opam_file in sorted(cwd.glob("*.opam")):
        try:
            text = opam_file.read_text()
        except OSError:
            continue
        parsed = parse_opam_file(text)
        for pd in parsed.get("pin_depends", []):
            pins.append({
                "package": pd["package"],
                "version": pd["version"],
                "url": pd["url"],
                "origin": "opam-file",
                "declared_in": opam_file.name,
            })
    return pins


def _pins_from_dune_project(cwd: Path) -> List[Dict[str, Any]]:
    """Extract (pin ...) stanzas from dune-project in cwd."""
    dune_project = cwd / "dune-project"
    if not dune_project.is_file():
        return []
    try:
        text = dune_project.read_text()
    except OSError:
        return []
    tree = parse_sexp(text)
    pin_stanzas = find_stanzas(tree, "pin")
    pins = []
    for stanza in pin_stanzas:
        # (pin (url ...) (package (name ...)) ...)
        # or simpler forms; extract what we can
        pin: Dict[str, Any] = {"origin": "dune-project"}
        for item in stanza[1:]:
            if isinstance(item, list) and len(item) >= 2:
                key = item[0]
                if key == "url":
                    pin["url"] = item[1] if isinstance(item[1], str) else str(item[1])
                elif key == "package":
                    # (package (name foo)) or (package (name foo) (version x))
                    for sub in item[1:]:
                        if isinstance(sub, list) and len(sub) >= 2:
                            if sub[0] == "name":
                                pin["package"] = sub[1]
                            elif sub[0] == "version":
                                pin["version"] = sub[1]
            elif isinstance(item, str) and "package" not in pin:
                # Some simple forms: (pin name url)
                if "package" not in pin:
                    pin["package"] = item
        pin.setdefault("package", "unknown")
        pin.setdefault("version", "dev")
        pins.append(pin)
    return pins


@mcp.tool(name="ocaml_deps_pins")
async def list_pins(switch: Optional[str] = None) -> Dict[str, Any]:
    """List all pinned OCaml packages from opam, .opam files, and dune-project.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Checks multiple sources and tags each pin with its origin.

    Args:
        switch: Optional opam switch name.

    Returns:
        List of pins with package, version, url, and origin fields
    """
    all_pins: List[Dict[str, Any]] = []
    cwd = Path.cwd()

    # opam pin list
    args = ["pin", "list"]
    if switch:
        args = ["--switch", switch] + args
    raw = await _run_opam(*args)
    if raw is not None:
        all_pins.extend(_parse_opam_pin_list(raw))

    # .opam files in cwd
    all_pins.extend(_pins_from_opam_files(cwd))

    # dune-project
    all_pins.extend(_pins_from_dune_project(cwd))

    return {"pins": all_pins, "total": len(all_pins)}


# ---------------------------------------------------------------------------
# Tool: ocaml_deps_repos
# ---------------------------------------------------------------------------

def _parse_opam_repo_list(raw: str) -> List[Dict[str, Any]]:
    """Parse output of 'opam repository list'."""
    repos = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: RANK NAME URL
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        try:
            rank = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        url = parts[2] if len(parts) > 2 else None
        repo: Dict[str, Any] = {"rank": rank, "name": name}
        if url:
            repo["url"] = url
        repos.append(repo)
    return repos


def _repos_from_dune_workspace(cwd: Path) -> tuple:
    """Extract repository definitions and ordering from dune-workspace.

    Returns (repo_list, repo_order) where repo_list has name/url dicts
    and repo_order is the list from (lock_dir (repositories ...)).
    """
    ws_file = cwd / "dune-workspace"
    if not ws_file.is_file():
        return [], []
    try:
        text = ws_file.read_text()
    except OSError:
        return [], []
    tree = parse_sexp(text)

    # Repository definitions: (repository (name x) (url y))
    repo_stanzas = find_stanzas(tree, "repository")
    repos = []
    for stanza in repo_stanzas:
        repo: Dict[str, Any] = {}
        for item in stanza[1:]:
            if isinstance(item, list) and len(item) >= 2:
                if item[0] == "name":
                    repo["name"] = item[1]
                elif item[0] == "url":
                    repo["url"] = item[1]
            elif isinstance(item, str) and "name" not in repo:
                repo["name"] = item
        if "name" in repo:
            repo.setdefault("url", None)
            repos.append(repo)

    # lock_dir ordering: (lock_dir (repositories repo1 repo2 ...))
    order = []
    lock_dir_stanzas = find_stanzas(tree, "lock_dir")
    for stanza in lock_dir_stanzas:
        for item in stanza[1:]:
            if isinstance(item, list) and len(item) >= 1 and item[0] == "repositories":
                order = [r for r in item[1:] if isinstance(r, str)]
                break
        if order:
            break

    return repos, order


@mcp.tool(name="ocaml_deps_repos")
async def list_repositories(switch: Optional[str] = None) -> Dict[str, Any]:
    """List configured OCaml package repositories from opam and dune-workspace.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Shows repository priority order so you can see which repos override others.

    Args:
        switch: Optional opam switch name.

    Returns:
        opam repositories (ranked) and dune-workspace repositories with ordering
    """
    result: Dict[str, Any] = {}
    cwd = Path.cwd()

    # opam repository list
    args = ["repository", "list"]
    if switch:
        args = ["--switch", switch] + args
    raw = await _run_opam(*args)
    if raw is not None:
        result["opam_repositories"] = _parse_opam_repo_list(raw)

    # dune-workspace
    dune_repos, dune_order = _repos_from_dune_workspace(cwd)
    if dune_repos:
        result["dune_repositories"] = dune_repos
    if dune_order:
        result["dune_repository_order"] = dune_order

    return result


# ---------------------------------------------------------------------------
# Tool: ocaml_deps_vendored
# ---------------------------------------------------------------------------

@mcp.tool(name="ocaml_deps_vendored")
async def list_vendored_dirs(path: Optional[str] = None) -> Dict[str, Any]:
    """Find vendored directories declared in OCaml dune files.

    ONLY for OCaml. Not useful for Rust, Python, JavaScript, or any other language.

    Searches dune files for vendored_dirs stanzas and checks for conventional
    vendor/ and duniverse/ directories.

    Args:
        path: Directory to search from. Defaults to current directory.

    Returns:
        Declared vendored dirs and any undeclared project directories
    """
    root = Path(path) if path else Path.cwd()
    if not root.is_dir():
        return {"error": f"Path does not exist: {root}"}

    vendored = []
    declared_abs = set()

    # Find vendored_dirs stanzas in dune files
    for dune_file in root.rglob("dune"):
        if not dune_file.is_file():
            continue
        try:
            text = dune_file.read_text()
        except OSError:
            continue
        tree = parse_sexp(text)
        stanzas = find_stanzas(tree, "vendored_dirs")
        for stanza in stanzas:
            for dir_name in stanza[1:]:
                if isinstance(dir_name, str):
                    abs_dir = (dune_file.parent / dir_name).resolve()
                    declared_abs.add(abs_dir)
                    vendored.append({
                        "dir": str(abs_dir),
                        "declared_in": str(dune_file),
                        "exists": abs_dir.is_dir(),
                    })

    # Check conventional directories
    undeclared = []
    for conventional in ("vendor", "duniverse"):
        conv_dir = (root / conventional).resolve()
        if conv_dir.is_dir() and conv_dir not in declared_abs:
            has_dune_project = (conv_dir / "dune-project").is_file()
            if has_dune_project:
                undeclared.append({
                    "dir": str(conv_dir),
                    "note": (
                        "Contains dune-project but not declared in vendored_dirs. "
                        "Dune will build it but won't suppress warnings."
                    ),
                })

    return {
        "vendored_dirs": vendored,
        "undeclared_project_dirs": undeclared,
    }


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
            register_doc_source("local", LocalDocSource(_local_docs_root))
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
                print("  search-packages <query> [repo_url]")
                print("  package-info <package> [version]")
                print("  module-doc <package> <module_path> [version]")
                print("  list-local")
                print("  local-module-doc <module_path>")
                print("  list-sources")
                print("  opam-versions <package>")
                print("  opam-show <package> [version]")
                print("  detect-dep-managers")
                print("  dep-env-status")
                print("  opam-installed")
                print("  list-pins")
                print("  list-repos")
                print("  list-vendored")
                return

            cmd = test_args[0]

            if cmd == "sherlodoc":
                query = test_args[1] if len(test_args) > 1 else "int -> string"
                result = await sherlodoc(query)
            elif cmd == "search-packages":
                query = test_args[1] if len(test_args) > 1 else "http"
                repos = [test_args[2]] if len(test_args) > 2 else None
                result = await search_package_names(query, repos)
            elif cmd == "package-info":
                pkg = test_args[1] if len(test_args) > 1 else "lwt"
                ver = test_args[2] if len(test_args) > 2 else None
                result = await get_package_info(pkg, ver)
            elif cmd == "module-doc":
                pkg = test_args[1] if len(test_args) > 1 else "lwt"
                mod = test_args[2] if len(test_args) > 2 else "Lwt"
                ver = test_args[3] if len(test_args) > 3 else None
                result = await get_module_doc(mod, package_name=pkg, version=ver)
            elif cmd == "list-local":
                result = await list_local_modules()
            elif cmd == "local-module-doc":
                mod = test_args[1] if len(test_args) > 1 else "Stdlib"
                result = await get_module_doc(mod, source="local")
            elif cmd == "opam-versions":
                pkg = test_args[1] if len(test_args) > 1 else "lwt"
                result = await opam_list_versions(pkg, OPAM_REPO_URL)
            elif cmd == "opam-show":
                pkg = test_args[1] if len(test_args) > 1 else "lwt"
                ver = test_args[2] if len(test_args) > 2 else None
                result = await opam_show_package(pkg, OPAM_REPO_URL, ver)
            elif cmd == "detect-dep-managers":
                result = await detect_dependency_managers()
            elif cmd == "dep-env-status":
                result = await dependency_environment_status()
            elif cmd == "opam-installed":
                source = test_args[1] if len(test_args) > 1 else "opam-switch"
                sw = test_args[2] if len(test_args) > 2 else None
                result = await list_installed_packages(source, sw)
            elif cmd == "list-pins":
                sw = test_args[1] if len(test_args) > 1 else None
                result = await list_pins(sw)
            elif cmd == "list-repos":
                sw = test_args[1] if len(test_args) > 1 else None
                result = await list_repositories(sw)
            elif cmd == "list-vendored":
                p = test_args[1] if len(test_args) > 1 else None
                result = await list_vendored_dirs(p)
            elif cmd == "list-sources":
                sources = [
                    {"name": s.name, "description": s.description, "priority": s.priority}
                    for s in sorted(_doc_sources.values(), key=lambda s: s.priority)
                ]
                result = {"sources": sources}
            else:
                result = {"error": f"Unknown command: {cmd}"}

            print(json.dumps(result, indent=2))

            # Clean up session
            global _session
            if _session and not _session.closed:
                await _session.close()

        asyncio.run(run_test())
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
