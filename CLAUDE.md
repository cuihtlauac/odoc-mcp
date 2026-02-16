# odoc-mcp

When responding keep the language accessible. It is offensive to use too complicated of language because it isn't inclusive of less educated individuals. Favour the simplest solution you can. All code in this project is AI-generated so feel free to be critical. You should avoid being overly agreeable and flattering in your responses.

## Overview

MCP server that gives LLMs access to OCaml package documentation and
dependency information. Documentation tools query
[sage.ci.dev](https://sage.ci.dev) and Sherlodoc. Dependency tools use the
local environment (opam CLI, `.opam` files, `dune-project`, `dune-workspace`,
dune build files) to show what's pinned, installed, vendored, and which
repositories are active.

## Development Environment

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
uv sync
uv run python mcp_server.py
```

## Project Structure

```
├── mcp_server.py       # MCP server (the main entry point)
├── opam_parser.py      # Parser for opam package files
├── sexp_parser.py      # Minimal S-expression parser for dune files
├── version_utils.py    # OCaml version string handling
├── pyproject.toml      # Project configuration and dependencies
├── uv.lock             # Dependency lock file
├── CLAUDE.md           # This file
└── README.md           # User-facing documentation
```

The remaining `.py` files in the repo are leftover from the old LLM pipeline
(extraction, embeddings, semantic search) and are not used by the MCP server.

## MCP Server (`mcp_server.py`)

Uses `mcp.server.fastmcp.FastMCP` with stdio transport (default) so Claude Code
can launch it directly.

### Available Tools

| Tool | Description |
|------|-------------|
| `ocaml_search` | Search by name or type signature across all packages |
| `ocaml_package_search` | Find packages by substring match (sage.ci.dev + opam repos) |
| `ocaml_package_doc` | Get package doc overview: description, libraries, and modules |
| `ocaml_module_doc` | Get a module's preamble and signatures (tries local, then sage.ci.dev) |
| `ocaml_module_list_local` | List modules in local odoc output |
| `ocaml_package_versions` | List all versions of a package |
| `ocaml_package_meta` | Show package metadata (deps, license, authors, etc.) |
| `ocaml_deps_managers` | List active dependency managers (opam, dune-pkg) |
| `ocaml_deps_status` | Report dependency environment details |
| `ocaml_deps_installed` | List installed packages from opam switch, dune-pkg, or system |
| `ocaml_deps_pins` | List pinned packages from opam, .opam files, and dune-project |
| `ocaml_deps_repos` | List configured repos from opam and dune-workspace |
| `ocaml_deps_vendored` | Find vendored directories declared in dune files |

### MCP Resources

| Resource | Description |
|----------|-------------|
| `ocaml-docs://sage` | Metadata for the sage.ci.dev documentation source (always available) |
| `ocaml-docs://local` | Metadata for the local odoc documentation source (when `--local-docs` is set) |

Clients discover available documentation sources via `resources/list`.
Each resource returns JSON with `name`, `description`, and `priority`
(lower priority number = tried first by `ocaml_module_doc`).

### Testing

```bash
uv run python mcp_server.py --test sherlodoc "List.map"
uv run python mcp_server.py --test search-packages lwt
uv run python mcp_server.py --test search-packages lwt https://github.com/ocaml/opam-repository
uv run python mcp_server.py --test package-info base
uv run python mcp_server.py --test module-doc base Base.List
uv run python mcp_server.py --local-docs _build/default/_doc/_html --test list-local
uv run python mcp_server.py --local-docs _build/default/_doc/_html --test local-module-doc MyModule
uv run python mcp_server.py --test list-sources
uv run python mcp_server.py --local-docs _build/default/_doc/_html --test list-sources
uv run python mcp_server.py --test opam-versions lwt
uv run python mcp_server.py --test opam-show lwt
uv run python mcp_server.py --test opam-show lwt 5.9.0
uv run python mcp_server.py --test detect-dep-managers
uv run python mcp_server.py --test dep-env-status
uv run python mcp_server.py --test opam-installed
uv run python mcp_server.py --test list-pins
uv run python mcp_server.py --test list-repos
uv run python mcp_server.py --test list-vendored
```

### Using with Claude Code

```bash
claude mcp add --scope user ocaml-docs -- uv run --directory /path/to/odoc-mcp python mcp_server.py
```

This makes the server available in all projects. To restrict it to the
current project, drop `--scope user`.

To include local docs (project-scoped, since the path is project-specific):

```bash
claude mcp add ocaml-docs -- uv run --directory /path/to/odoc-mcp python mcp_server.py --local-docs /path/to/_build/default/_doc/_html
```

### Using with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ocaml-docs": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/odoc-mcp"
    }
  }
}
```
