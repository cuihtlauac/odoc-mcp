#!/usr/bin/env python3
"""
Generate embeddings for OCaml package descriptions.

This script reads package descriptions from package-descriptions/ and generates
embeddings using a local embedding model server.
"""

import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False

def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

@dataclass
class PackageDescription:
    """Package description data."""
    name: str
    version: str
    description: str

@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    package_name: str
    embedding: np.ndarray
    success: bool
    error: Optional[str] = None

class EmbeddingClient:
    """Client for generating embeddings using a local embedding server."""
    
    def __init__(self, base_url: str = "http://localhost:8000", model: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.base_url = base_url
        self.model = model
        self.embedding_url = f"{base_url}/embedding"
        print(f"EmbeddingClient initialized with base_url={base_url}, model={model}")
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a text string."""
        try:
            payload = {
                "model": self.model,
                "input": text
            }
            
            response = requests.post(
                self.embedding_url,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                print(f"API returned status {response.status_code}: {response.text[:200]}")
                return None
                
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                # Check if it's a list of dicts (OpenAI-style)
                if isinstance(result[0], dict) and 'embedding' in result[0]:
                    raw_embedding = np.array(result[0]['embedding'], dtype=np.float32)
                elif isinstance(result[0], (list, np.ndarray)):
                    # Response is a list of embeddings
                    raw_embedding = np.array(result[0], dtype=np.float32)
                else:
                    return None
            elif isinstance(result, dict) and 'data' in result and len(result['data']) > 0:
                # OpenAI-style response format
                raw_embedding = np.array(result['data'][0]['embedding'], dtype=np.float32)
            else:
                return None
                
            # Handle multi-dimensional embeddings by taking the mean
            if raw_embedding.ndim == 2:
                # If we have a 2D array (sequence_length, embedding_dim), take the mean
                embedding = np.mean(raw_embedding, axis=0)
            elif raw_embedding.ndim == 1:
                # If we have a 1D array, use it directly
                embedding = raw_embedding
            else:
                return None
                
            # Validate embedding
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                return None
                
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
                
        except requests.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

def load_package_descriptions(input_dir: Path) -> List[PackageDescription]:
    """Load all package descriptions from JSON files."""
    descriptions = []
    
    for json_file in input_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'package' in data and 'description' in data:
                descriptions.append(PackageDescription(
                    name=data['package'],
                    version=data.get('version', 'unknown'),
                    description=data['description']
                ))
            else:
                pass  # Skip invalid files silently
                
        except Exception:
            pass  # Skip files with errors silently
            
    return descriptions

def generate_package_embedding(package: PackageDescription, client: EmbeddingClient) -> EmbeddingResult:
    """Generate embedding for a single package description."""
    global shutdown_requested
    
    if shutdown_requested:
        return EmbeddingResult(package.name, None, False, "Shutdown requested")
    
    try:
        # Create embedding text that includes package name and description
        embedding_text = f"Package: {package.name}\nDescription: {package.description}"
        
        embedding = client.generate_embedding(embedding_text)
        
        if embedding is not None:
            return EmbeddingResult(package.name, embedding, True)
        else:
            return EmbeddingResult(package.name, None, False, "Failed to generate embedding")
            
    except Exception as e:
        return EmbeddingResult(package.name, None, False, f"Exception: {str(e)}")

def process_packages_batch(packages: List[PackageDescription], client: EmbeddingClient, 
                          batch_size: int = 16) -> List[EmbeddingResult]:
    """Process a batch of packages with rate limiting."""
    results = []
    
    for i in range(0, len(packages), batch_size):
        if shutdown_requested:
            break
            
        batch = packages[i:i + batch_size]
        batch_results = []
        
        # Process batch with some parallelization
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            future_to_package = {
                executor.submit(generate_package_embedding, package, client): package
                for package in batch
            }
            
            for future in as_completed(future_to_package):
                if shutdown_requested:
                    break
                result = future.result()
                batch_results.append(result)
        
        results.extend(batch_results)
        
        # Rate limiting between batches
        if i + batch_size < len(packages) and not shutdown_requested:
            time.sleep(0.1)  # Small delay between batches
    
    return results

def save_embeddings(results: List[EmbeddingResult], output_dir: Path) -> None:
    """Save embeddings in per-package structure like module embeddings."""
    successful_results = [r for r in results if r.success]
    
    if not successful_results:
        print("No successful embeddings to save")
        return
    
    # Check embedding dimensions
    embedding_dims = [r.embedding.shape for r in successful_results]
    unique_dims = set(embedding_dims)
    
    if len(unique_dims) > 1:
        print(f"Error: Embeddings have inconsistent dimensions: {unique_dims}")
        # Find which packages have different dimensions
        for i, (result, dim) in enumerate(zip(successful_results[:10], embedding_dims[:10])):
            print(f"  {result.package_name}: {dim}")
        return
    
    # Create packages directory
    packages_dir = output_dir / "packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each package separately
    for result in successful_results:
        package_dir = packages_dir / result.package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embedding
        embeddings_file = package_dir / "embeddings.npz"
        np.savez_compressed(embeddings_file, embeddings=result.embedding[None, :])  # Add batch dimension
        
        # Save package metadata
        metadata = {
            "package": result.package_name,
            "embedding_dimension": result.embedding.shape[0],
            "generation_time": time.time()
        }
        
        metadata_file = package_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    # Save global metadata
    global_metadata = {
        "total_packages": len(successful_results),
        "embedding_dimension": successful_results[0].embedding.shape[0],
        "packages": [r.package_name for r in successful_results],
        "generation_time": time.time()
    }
    
    global_metadata_file = output_dir / "metadata.json"
    with open(global_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(global_metadata, f, indent=2)
    
    print(f"Saved {len(successful_results)} package embeddings to {packages_dir}")
    print(f"Saved global metadata to {global_metadata_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for OCaml package descriptions")
    parser.add_argument("--input-dir", default="package-descriptions", 
                       help="Directory containing package description JSON files")
    parser.add_argument("--output-dir", default="package-embeddings", 
                       help="Output directory for embeddings")
    parser.add_argument("--embedding-url", default="http://localhost:8000",
                       help="Base URL for embedding server")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                       help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--limit", type=int, 
                       help="Limit number of packages to process (for testing)")
    parser.add_argument("--packages", nargs="+",
                       help="Process specific packages only")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load package descriptions
    print(f"Loading package descriptions from {input_dir}")
    packages = load_package_descriptions(input_dir)
    
    if not packages:
        print("No package descriptions found")
        return 1
    
    print(f"Found {len(packages)} package descriptions")
    
    # Filter by specific packages if specified
    if args.packages:
        packages = [p for p in packages if p.name in args.packages]
        print(f"Filtered to {len(packages)} packages: {args.packages}")
    
    # Apply limit if specified
    if args.limit:
        packages = packages[:args.limit]
        print(f"Limited to {len(packages)} packages")
    
    if not packages:
        print("No packages to process")
        return 0
    
    # Initialize embedding client
    try:
        client = EmbeddingClient(args.embedding_url, args.model)
    except Exception as e:
        print(f"Failed to initialize embedding client: {e}")
        return 1
    
    # Generate embeddings
    print(f"Generating embeddings for {len(packages)} packages")
    start_time = time.time()
    
    results = process_packages_batch(packages, client, args.batch_size)
    
    if shutdown_requested:
        print("Processing interrupted by shutdown signal")
        return 1
    
    # Report results
    successful = len([r for r in results if r.success])
    failed = len([r for r in results if not r.success])
    
    elapsed = time.time() - start_time
    print(f"Processing complete in {elapsed:.1f}s")
    print(f"Successfully processed: {successful}/{len(packages)} packages")
    
    if failed > 0:
        print(f"Failed to process {failed} packages")
        # Log first few failures for debugging
        for result in results[:5]:
            if not result.success:
                print(f"Failed {result.package_name}: {result.error}")
    
    # Save embeddings
    if successful > 0:
        save_embeddings(results, output_dir)
        print(f"Package embeddings saved to {output_dir}")
    else:
        print("No successful embeddings to save")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())