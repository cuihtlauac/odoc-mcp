#!/usr/bin/env python3
"""
FastMCP Server for OCaml Documentation Search

This server exposes tools for searching and interacting with the OCaml
documentation dataset through the Model Context Protocol (MCP) using HTTP SSE.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

# Import semantic search components
from semantic_search_fixed import SemanticSearchEngine
from unified_search_all import UnifiedSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("ocaml-search", host="0.0.0.0")

# Global search engine instances (lazy loaded)
search_engine: Optional[SemanticSearchEngine] = None
unified_engine: Optional[UnifiedSearchEngine] = None
embeddings_dir = Path("package-embeddings")
package_descriptions_dir = Path("package-descriptions")
indexes_dir = Path("module-indexes")
module_descriptions_dir = Path("module-descriptions")

@mcp.tool()
async def find_ocaml_packages(functionality: str, popularity_weight: float = 0.3) -> Dict[str, Any]:
    """
    Discover OCaml packages across the entire ecosystem for specific functionality.
    
    Use this when you don't know which packages might contain what you need.
    This searches across all available packages to find the most relevant ones.
    
    Args:
        functionality: Describe what you're looking for. Be specific:
                      - "WebSocket client implementation"
                      - "Machine learning matrix operations" 
                      - "CSV file parsing and writing"
                      - "OAuth2 authentication flow"
                      - "Image processing and filtering"
        popularity_weight: Weight for popularity in ranking (0.0-1.0, default: 0.3)
    
    Returns:
        List of packages ranked by relevance, each with:
        - package: Package name
        - module: Primary module providing the functionality  
        - description: What the module does
    """
    global search_engine
    
    # Initialize search engine on first use (lazy loading)
    if search_engine is None:
        logger.info("Initializing semantic search engine...")
        try:
            search_engine = SemanticSearchEngine(
                module_embeddings_dir=embeddings_dir,
                package_embeddings_dir=Path("package-description-embeddings"),
                api_url="http://localhost:8080"
            )
        except Exception as e:
            error_msg = f"Failed to initialize search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Perform semantic search with popularity
        results = search_engine.search(functionality, top_k=5, 
                                     popularity_weight=popularity_weight)
        
        # Format results for MCP response
        return {
            "query": functionality,
            "packages": [
                {
                    "package": result["package"],
                    "library": result.get("library"),
                    "module": result["module_path"],
                    "description": result["description"]
                }
                for result in results
            ]
        }
        
    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def get_package_summary(package_name: str) -> Dict[str, Any]:
    """
    Get a concise overview of what an OCaml package does.
    
    Provides a 3-4 sentence summary explaining the package's purpose, main features,
    and typical use cases. Helpful for understanding a package before using it.
    
    Args:
        package_name: The OCaml package name you want to learn about:
                     - 'lwt' (asynchronous programming)
                     - 'base' (alternative standard library)  
                     - 'cohttp' (HTTP client/server)
                     - 'cmdliner' (command-line interfaces)
                     - 'yojson' (JSON processing)
    
    Returns:
        Package name, version, and description explaining what it does
    """
    try:
        description_file = package_descriptions_dir / f"{package_name}.json"
        
        if not description_file.exists():
            return {
                "error": f"No description found for package '{package_name}'. Package may not exist or description not yet generated."
            }
        
        with open(description_file, 'r', encoding='utf-8') as f:
            package_data = json.load(f)
        
        return {
            "package": package_data["package"],
            "version": package_data["version"],
            "description": package_data["description"]
        }
        
    except Exception as e:
        error_msg = f"Failed to get package summary: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}




@mcp.tool()
async def search_ocaml_modules(query: str, packages: Optional[List[str]] = None, top_k: int = 8, 
                              popularity_weight: float = 0.3) -> Dict[str, Any]:
    """
    Find OCaml modules that provide specific functionality across the OCaml ecosystem.
    
    Provide a clear description of the functionality you need. The tool will search across
    all OCaml packages by default, or within specific packages if you specify them.
    Uses both conceptual understanding and exact keyword matching.
    
    Args:
        query: Specific functionality you're looking for. Be precise about what you need:
               - "MD5 hash function" 
               - "HTTP client for making requests"
               - "JSON parsing and serialization"
               - "list sorting operations"
               - "TCP socket server"
               - "date time parsing and formatting"
        packages: Optional list of specific packages to search within. If not provided,
                 searches all available packages. Common packages include:
                 ['base', 'core', 'lwt', 'async', 'cohttp', 'yojson', 'cmdliner']
        top_k: Maximum number of results to return (default: 8)
        popularity_weight: Weight for popularity in ranking (0.0-1.0, default: 0.3)
    
    Returns:
        Two lists of matching modules:
        - semantic_results: Modules with conceptually similar functionality (includes descriptions)
        - keyword_results: Modules containing your exact keywords in their documentation
        
        Each result includes the package name, module name, and module path.
    """
    global unified_engine
    
    # Initialize unified search engine on first use (lazy loading)
    if unified_engine is None:
        logger.info("Initializing unified search engine...")
        try:
            unified_engine = UnifiedSearchEngine(
                embedding_dir=embeddings_dir,
                index_dir=indexes_dir,
                api_url="http://localhost:8080"
            )
        except Exception as e:
            error_msg = f"Failed to initialize unified search engine: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    try:
        # Load specified packages and perform search
        if packages:
            unified_engine.load_package_data(packages)
        else:
            unified_engine.load_package_data(None)  # Load all packages
            
        results = unified_engine.search(query, top_k)
        
        # Format results for MCP response
        return {
            "query": query,
            "packages_searched": packages if packages else "all",
            "semantic_results": [
                {
                    "package": r["package"],
                    "library": r.get("library", ""),
                    "module": r["module_path"],
                    "description": r["description"],
                    "score": r["score"]
                }
                for r in results["semantic"]
            ],
            "keyword_results": [
                {
                    "package": r["package"],
                    "library": r.get("library", ""),
                    "module": r["module_path"],
                    "description": r["description"],
                    "score": r["score"]
                }
                for r in results["keyword"]
            ],
            "total_results": len(results["semantic"]) + len(results["keyword"])
        }
        
    except Exception as e:
        error_msg = f"Unified search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
async def sherlodoc(query: str) -> Dict[str, Any]:
    """
    Search OCaml documentation using Sherlodoc - particularly effective for type searches.
    
    Sherlodoc is specialized for finding types, functions, and modules across all OCaml packages.
    It excels at type-based queries and can find exact matches for complex type signatures.
    
    Args:
        query: Your search query. Can be:
               - Type signatures: "int -> string -> bool"
               - Module paths: "Base.List.t"
               - Function names: "List.map"
               - Type definitions: "result"
               - Complex types: "('a -> 'b) -> 'a list -> 'b list"
    
    Returns:
        Search results including type definitions, functions, and their documentation
    """
    try:
        # URL encode the query
        encoded_query = quote(query)
        url = f"http://localhost:1234/api?q={encoded_query}"
        
        # Make async HTTP request
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return {"error": f"Sherlodoc API returned status {response.status}"}
                
                html_content = await response.text()
        
        # Parse HTML response
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract query info
        query_div = soup.find('div', class_='query')
        query_text = query_div.get_text(strip=True) if query_div else f"Results for {query}"
        
        # Extract search results
        results = []
        found_items = soup.find_all('li')
        
        for item in found_items[:20]:  # Limit to first 20 results
            result = {}
            
            # Extract package info
            pkg_div = item.find('div', class_='pkg')
            if pkg_div and pkg_div.find('a'):
                # Note: The HTML seems to have empty package info, we'll work with what we have
                result['package'] = 'Unknown'  # Package info not provided in the HTML
            
            # Extract the main code/signature
            pre_tag = item.find('pre')
            if pre_tag:
                # Get the full text including emphasized parts
                signature_parts = []
                for elem in pre_tag.descendants:
                    if elem.name is None:  # Text node
                        signature_parts.append(str(elem))
                    elif elem.name == 'em':
                        signature_parts.append(elem.get_text())
                
                result['signature'] = ''.join(signature_parts).strip()
                
                # Extract the link if available
                link_tag = pre_tag.find('a')
                if link_tag and link_tag.get('href'):
                    result['url'] = link_tag['href']
                    # Extract module path from the link text
                    em_tag = link_tag.find('em')
                    if em_tag:
                        result['module_path'] = em_tag.get_text()
            
            # Extract documentation/comment
            comment_div = item.find('div', class_='comment')
            if comment_div:
                # Extract text from all paragraphs
                doc_parts = []
                for p in comment_div.find_all('p'):
                    doc_parts.append(p.get_text(strip=True))
                if doc_parts:
                    result['documentation'] = ' '.join(doc_parts)
            
            if result and 'signature' in result:
                results.append(result)
        
        return {
            "query": query,
            "query_info": query_text,
            "results": results,
            "total_results": len(results)
        }
        
    except aiohttp.ClientError as e:
        error_msg = f"Failed to connect to Sherlodoc API: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Sherlodoc search failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def main():
    """Main entry point for the FastMCP server."""
    import sys
    import asyncio
    
    # Check for --test flag for direct testing
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run a simple test
        async def test():
            if len(sys.argv) > 2 and sys.argv[2] == "--summary":
                # Test package summary functionality
                package = sys.argv[3] if len(sys.argv) > 3 else "lwt"
                print(f"Testing package summary for: {package}\n")
                result = await get_package_summary(package)
                print(json.dumps(result, indent=2))
            elif "--packages" in sys.argv:
                # Test unified search functionality
                query_idx = sys.argv.index("--test") + 1
                packages_idx = sys.argv.index("--packages") + 1
                query = sys.argv[query_idx] if query_idx < len(sys.argv) else "HTTP server"
                packages = sys.argv[packages_idx:] if packages_idx < len(sys.argv) else ["base", "lwt", "cohttp"]
                print(f"Testing unified search query: {query}")
                print(f"Packages: {packages}\n")
                result = await search_ocaml_modules(query, packages)
                print(json.dumps(result, indent=2))
            elif "--sherlodoc" in sys.argv:
                # Test sherlodoc functionality
                sherlodoc_idx = sys.argv.index("--sherlodoc")
                query = sys.argv[sherlodoc_idx + 1] if sherlodoc_idx + 1 < len(sys.argv) else "Base.List.t"
                print(f"Testing sherlodoc query: {query}\n")
                result = await sherlodoc(query)
                print(json.dumps(result, indent=2))
            else:
                # Test search functionality
                query = sys.argv[2] if len(sys.argv) > 2 else "HTTP server"
                print(f"Testing search query: {query}\n")
                result = await find_ocaml_packages(query)
                print(json.dumps(result, indent=2))
        
        asyncio.run(test())
    else:
        # Run FastMCP server with HTTP SSE transport
        # Default SSE endpoint will be available at /sse
        # Using default port (probably 8000), may conflict with embedding server
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()