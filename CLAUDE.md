# odoc-mcp

When responding keep the language accessible. It is offensive to use too complicated of language because it isn't inclusive of less educated individuals. Favour the simplest solution you can. All code in this project is AI-generated so feel free to be critical. You should avoid being overly agreeable and flattering in your responses.

## Overview

MCP server that gives LLMs access to OCaml package documentation. It queries
[sage.ci.dev](https://sage.ci.dev) for any published package and can also
browse locally-built odoc output (from `dune build @doc`).

## Development Environment

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
uv sync
uv run python mcp_server.py
```

## Project Structure

```
├── mcp_server.py       # MCP server (the main entry point)
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
| `sherlodoc` | Search by name or type signature across all packages |
| `search_package_names` | Find packages by substring match |
| `get_package_info` | Get package description, libraries, and module list |
| `get_module_doc` | Get a module's preamble and signatures from sage.ci.dev |
| `list_local_modules` | List modules in local odoc output |
| `get_local_module_doc` | Get a module's preamble and signatures from local docs |

### Testing

```bash
uv run python mcp_server.py --test sherlodoc "List.map"
uv run python mcp_server.py --test search-packages lwt
uv run python mcp_server.py --test package-info base
uv run python mcp_server.py --test module-doc base Base.List
uv run python mcp_server.py --local-docs _build/default/_doc/_html --test list-local
uv run python mcp_server.py --local-docs _build/default/_doc/_html --test local-module-doc MyModule
```

### Using with Claude Code

```bash
claude mcp add ocaml-docs -- uv run --directory /path/to/odoc-mcp python mcp_server.py
```

To include local docs:

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
