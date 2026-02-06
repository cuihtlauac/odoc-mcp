# odoc-mcp

MCP server that gives LLMs access to OCaml package documentation. It can query
[sage.ci.dev](https://sage.ci.dev) for any published package and also browse
locally-built odoc output (from `dune build @doc`).

## Why

LLMs don't know the OCaml ecosystem well. This server lets them look up
module signatures, read preambles, and search packages by name or type
signature — so they can write better OCaml code.

## Using with Claude Code

```bash
claude mcp add --scope user ocaml-docs -- uv run --directory /path/to/odoc-mcp python mcp_server.py
```

This makes the server available in all projects. To restrict it to the
current project, drop `--scope user`.

To include local docs (project-scoped, since the path is project-specific):

```bash
claude mcp add ocaml-docs -- uv run --directory /path/to/odoc-mcp python mcp_server.py --local-docs /path/to/_build/default/_doc/_html
```

## Using with Claude Desktop

Add this to `claude_desktop_config.json`:

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

To include local docs, add `"--local-docs", "/path/to/_build/default/_doc/_html"`
to the `args` array.

### Available tools

| Tool | Description |
|------|-------------|
| `sherlodoc` | Search by name or type signature across all packages |
| `search_package_names` | Find packages by substring match |
| `get_package_info` | Get package description, libraries, and module list |
| `get_module_doc` | Get a module's preamble and signatures from sage.ci.dev |
| `list_local_modules` | List modules in local odoc output |
| `get_local_module_doc` | Get a module's preamble and signatures from local docs |

## Testing

```bash
uv run python mcp_server.py --test sherlodoc "List.map"
uv run python mcp_server.py --test search-packages lwt
uv run python mcp_server.py --test package-info base
uv run python mcp_server.py --test module-doc base Base.List
uv run python mcp_server.py --local-docs _build/default/_doc/_html --test list-local
uv run python mcp_server.py --local-docs _build/default/_doc/_html --test local-module-doc MyModule
```
