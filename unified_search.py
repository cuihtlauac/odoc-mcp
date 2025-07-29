#!/usr/bin/env python3
"""
Unified search for OCaml modules using both BM25 and embedding search.

This script combines sparse (BM25) and dense (embedding) retrieval methods
to provide comprehensive search results across OCaml module documentation.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import bm25s
from tqdm import tqdm
import requests

from popularity_scorer import PopularityScorer


class UnifiedSearchEngine:
    def __init__(self, embedding_dir: str = "package-embeddings", 
                 index_dir: str = "module-indexes",
                 api_url: str = 'http://localhost:8080',
                 use_popularity: bool = True):
        self.embedding_dir = Path(embedding_dir)
        self.index_dir = Path(index_dir)
        self.embeddings = {}
        self.metadata = {}
        self.bm25_indexes = {}
        self.module_paths = {}
        
        # Initialize popularity scorer
        self.use_popularity = use_popularity
        self.popularity_scorer = PopularityScorer() if use_popularity else None
        
        # Initialize embedding API
        self.api_url = api_url
        self.embedding_url = f"{api_url}/embedding"
        print(f"Using embedding API at: {self.embedding_url}")
        
        # Test the API
        try:
            response = requests.get(api_url + "/health", timeout=5)
            if response.status_code == 200:
                print("API health check passed")
        except:
            print("API health check failed - continuing anyway")
            
        self.max_length = 8192
        
    def load_package_data(self, package_names: List[str]) -> None:
        """Load embeddings and BM25 indexes for specified packages."""
        print(f"Loading data for {len(package_names)} packages...")
        
        for package in tqdm(package_names, desc="Loading packages"):
            # Load embeddings
            embedding_path = self.embedding_dir / "packages" / package / "embeddings.npz"
            metadata_path = self.embedding_dir / "packages" / package / "metadata.json"
            
            if embedding_path.exists() and metadata_path.exists():
                # Load embeddings
                data = np.load(embedding_path)
                package_embeddings = data['embeddings']
                
                # Load metadata
                with open(metadata_path, 'r') as f:
                    package_metadata = json.load(f)
                
                # Store with package prefix in module paths
                for i, module_info in enumerate(package_metadata['modules']):
                    full_path = f"{package}::{module_info['module_path']}"
                    self.embeddings[full_path] = package_embeddings[i]
                    metadata_entry = {
                        'package': package,
                        'module_path': module_info['module_path'],
                        'description': module_info.get('description', '')
                    }
                    # Include library information if available
                    if 'library' in module_info:
                        metadata_entry['library'] = module_info['library']
                    self.metadata[full_path] = metadata_entry
            
            # Load BM25 index
            index_path = self.index_dir / package
            if index_path.exists():
                # Load the BM25 index
                retriever = bm25s.BM25.load(str(index_path / "index"))
                self.bm25_indexes[package] = retriever
                
                # Load module paths for this package
                with open(index_path / "module_paths.json", 'r') as f:
                    self.module_paths[package] = json.load(f)
    
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """Get embedding for a query string using llama-server API."""
        try:
            payload = {"content": query}
            
            response = requests.post(
                self.embedding_url,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"API returned status {response.status_code}: {response.text[:200]}")
                
            result = response.json()
            
            # Handle llama-server response format
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'embedding' in result[0]:
                    # Extract the embedding (it's nested)
                    embedding_data = result[0]['embedding']
                    if isinstance(embedding_data, list) and len(embedding_data) > 0:
                        embedding = np.array(embedding_data[0], dtype=np.float32)
                    else:
                        embedding = np.array(embedding_data, dtype=np.float32)
                else:
                    raise Exception(f"Unexpected response format: {result[:200]}")
            else:
                raise Exception(f"Unexpected response format: {result[:200]}")
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
            return embedding
            
        except Exception as e:
            print(f"Failed to generate embedding: {e}")
            raise
    
    def embedding_search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Perform embedding-based semantic search."""
        if not self.embeddings:
            return []
        
        # Get query embedding
        query_embedding = self.get_query_embedding(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Calculate similarities
        results = []
        for module_path, module_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, module_embedding)
            results.append((module_path, similarity, self.metadata[module_path]))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def bm25_search(self, query: str, packages: List[str], top_k: int = 10) -> List[Tuple[str, float, str]]:
        """Perform BM25 search across specified packages."""
        all_results = []
        
        for package in packages:
            if package not in self.bm25_indexes:
                continue
            
            retriever = self.bm25_indexes[package]
            module_paths = self.module_paths[package]
            
            # Tokenize query
            query_tokens = bm25s.tokenize([query], stopwords="en")
            
            # Search
            results, scores = retriever.retrieve(query_tokens, k=min(top_k, len(module_paths)))
            
            # Convert results to module paths with scores
            for doc_ids, doc_scores in zip(results, scores):
                for doc_id, score in zip(doc_ids, doc_scores):
                    if 0 <= doc_id < len(module_paths):
                        full_path = f"{package}::{module_paths[doc_id]}"
                        all_results.append((full_path, float(score), package))
        
        # Sort all results by score and return top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    def unified_search(self, query: str, packages: List[str], top_k: int = 10, 
                      popularity_weight: float = 0.3) -> Dict:
        """Perform unified search combining BM25 and embedding results."""
        # Load data for specified packages if not already loaded
        packages_to_load = [p for p in packages if p not in self.bm25_indexes]
        if packages_to_load:
            self.load_package_data(packages_to_load)
        
        # Perform both searches (get more results for reranking)
        initial_top_k = min(top_k * 3, 30)  # Get 3x results for reranking
        embedding_results = self.embedding_search(query, initial_top_k)
        bm25_results = self.bm25_search(query, packages, initial_top_k)
        
        # Filter embedding results to only include specified packages
        embedding_results = [
            (path, score, meta) for path, score, meta in embedding_results
            if meta['package'] in packages
        ]
        
        # Apply popularity reranking if enabled
        if self.use_popularity and self.popularity_scorer:
            # Rerank embedding results
            embedding_tuples = [(meta['package'], meta['module_path'], 
                               meta.get('description', ''), score) 
                              for path, score, meta in embedding_results]
            reranked_embedding = self.popularity_scorer.rerank_results(
                embedding_tuples, popularity_weight)
            
            # Rerank BM25 results
            bm25_tuples = []
            for path, score, package in bm25_results:
                module_name = path.split('::')[1] if '::' in path else path
                # Get description from metadata if available
                description = ''
                if path in self.metadata:
                    description = self.metadata[path].get('description', '')
                bm25_tuples.append((package, module_name, description, score))
            reranked_bm25 = self.popularity_scorer.rerank_results(
                bm25_tuples, popularity_weight)
        else:
            # No popularity scoring, use original results
            reranked_embedding = [(meta['package'], meta['module_path'], 
                                 meta.get('description', ''), score) 
                                for path, score, meta in embedding_results]
            reranked_bm25 = []
            for path, score, package in bm25_results:
                module_name = path.split('::')[1] if '::' in path else path
                description = self.metadata.get(path, {}).get('description', '')
                reranked_bm25.append((package, module_name, description, score))
        
        # Deduplicate and format results
        seen_modules = set()
        deduplicated_embedding = []
        deduplicated_bm25 = []
        
        # Process reranked embedding results
        for package, module_path, description, score in reranked_embedding[:top_k]:
            module_full_path = f"{package}::{module_path}"
            if module_full_path not in seen_modules:
                seen_modules.add(module_full_path)
                deduplicated_embedding.append({
                    'module_path': module_full_path,
                    'score': float(score),
                    'package': package,
                    'module_name': module_path,
                    'description': description
                })
        
        # Process reranked BM25 results
        for package, module_name, description, score in reranked_bm25[:top_k]:
            bm25_full_path = f"{package}::{module_name}"
            if bm25_full_path not in seen_modules:
                seen_modules.add(bm25_full_path)
                deduplicated_bm25.append({
                    'module_path': bm25_full_path,
                    'score': float(score),
                    'package': package,
                    'module_name': module_name,
                    'description': description
                })
        
        return {
            'query': query,
            'packages_searched': packages,
            'embedding_results': deduplicated_embedding,
            'bm25_results': deduplicated_bm25
        }


def main():
    parser = argparse.ArgumentParser(description="Unified search for OCaml modules")
    parser.add_argument("query", help="Search query")
    parser.add_argument("packages", nargs="+", help="Packages to search in")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--embedding-dir", default="package-embeddings",
                        help="Directory containing package embeddings")
    parser.add_argument("--index-dir", default="module-indexes",
                        help="Directory containing BM25 indexes")
    parser.add_argument("--api-url", default="http://localhost:8080",
                        help="llama-server API URL")
    parser.add_argument("--format", choices=["text", "json"], default="text",
                        help="Output format")
    parser.add_argument("--no-popularity", action="store_true",
                        help="Disable popularity-based ranking")
    parser.add_argument("--popularity-weight", type=float, default=0.3,
                        help="Weight for popularity in ranking (0.0-1.0, default: 0.3)")
    
    args = parser.parse_args()
    
    # Create search engine
    engine = UnifiedSearchEngine(
        embedding_dir=args.embedding_dir,
        index_dir=args.index_dir,
        api_url=args.api_url,
        use_popularity=not args.no_popularity
    )
    
    # Perform search
    results = engine.unified_search(args.query, args.packages, args.top_k, 
                                  popularity_weight=args.popularity_weight)
    
    if args.format == "json":
        print(json.dumps(results, indent=2))
    else:
        print(f"\n🔍 Search results for: '{args.query}'")
        print(f"📦 Packages searched: {', '.join(args.packages)}")
        
        print(f"\n🧠 Semantic Search Results (Embeddings):")
        print("─" * 80)
        for i, result in enumerate(results['embedding_results'], 1):
            print(f"{i}. {result['package']}::{result['module_name']} (score: {result['score']:.4f})")
            if result['description']:
                print(f"   {result['description'][:150]}...")
            print()
        
        print(f"\n📝 Full-Text Search Results (BM25):")
        print("─" * 80)
        for i, result in enumerate(results['bm25_results'], 1):
            print(f"{i}. {result['package']}::{result['module_name']} (score: {result['score']:.4f})")
            print()


if __name__ == "__main__":
    main()