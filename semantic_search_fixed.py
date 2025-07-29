#!/usr/bin/env python3
"""
Fixed Semantic Search for OCaml Modules using llama-server API

This script embeds user queries using the same llama-server API that was used
to generate the module embeddings, ensuring consistency.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
import requests

from popularity_scorer import PopularityScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QueryEmbedder:
    """Handles embedding of user queries using llama-server API."""
    
    def __init__(self, api_url: str = 'http://localhost:8080'):
        """Initialize the query embedder with the API endpoint."""
        self.api_url = api_url
        self.embedding_url = f"{api_url}/embedding"
        logger.info(f"Using embedding API at: {self.embedding_url}")
        
        # Test the API
        try:
            response = requests.get(api_url + "/health", timeout=5)
            if response.status_code == 200:
                logger.info("API health check passed")
        except:
            logger.warning("API health check failed - continuing anyway")
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query using llama-server API.
        
        Args:
            query: User's search query
            
        Returns:
            Normalized embedding vector
        """
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
            logger.error(f"Failed to generate embedding: {e}")
            raise


class PackageEmbeddingLoader:
    """Handles loading of pre-computed module and package embeddings."""
    
    def __init__(self, module_embeddings_dir: Path, package_embeddings_dir: Path = None):
        """Initialize the loader with embedding directories."""
        self.module_embeddings_dir = module_embeddings_dir
        self.module_packages_dir = module_embeddings_dir / "packages"
        
        # Package embeddings directory (for package descriptions)
        self.package_embeddings_dir = package_embeddings_dir
        if self.package_embeddings_dir:
            self.package_packages_dir = self.package_embeddings_dir / "packages"
        else:
            self.package_packages_dir = None
        
        # Check directories exist
        if not self.module_packages_dir.exists():
            raise ValueError(f"Module embeddings directory not found: {self.module_packages_dir}")
        
        if self.package_packages_dir and not self.package_packages_dir.exists():
            logger.warning(f"Package embeddings directory not found: {self.package_packages_dir}")
            self.package_packages_dir = None
    
    def load_all_embeddings(self) -> Tuple[np.ndarray, List[Dict], Dict[str, np.ndarray]]:
        """
        Load all module embeddings and package description embeddings into memory.
        
        Returns:
            Tuple of (module_embeddings, module_metadata, package_embeddings_dict)
        """
        all_embeddings = []
        all_metadata = []
        package_embeddings = {}
        
        # Find all packages with module embeddings
        package_dirs = sorted([d for d in self.module_packages_dir.iterdir() if d.is_dir()])
        logger.info(f"Found {len(package_dirs)} packages with embeddings")
        
        # Load module embeddings
        for package_dir in package_dirs:
            try:
                # Load embeddings
                embeddings_file = package_dir / "embeddings.npz"
                if not embeddings_file.exists():
                    continue
                    
                data = np.load(embeddings_file)
                embeddings = data['embeddings']
                
                # Load metadata
                metadata_file = package_dir / "metadata.json"
                if not metadata_file.exists():
                    continue
                    
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # Extract module information
                package_name = metadata['package']
                modules = metadata.get('modules', [])
                
                if len(embeddings) != len(modules):
                    logger.warning(f"Mismatch in {package_name}: {len(embeddings)} embeddings vs {len(modules)} modules")
                    continue
                
                # Store embeddings and metadata with package name
                all_embeddings.append(embeddings)
                # Add package name to each module's metadata
                for module in modules:
                    module['package'] = package_name
                all_metadata.extend(modules)
                
            except Exception as e:
                logger.error(f"Error loading embeddings from {package_dir}: {e}")
                continue
        
        # Load package description embeddings if available
        if self.package_packages_dir:
            logger.info("Loading pre-computed package description embeddings...")
            for package_dir in self.package_packages_dir.iterdir():
                if not package_dir.is_dir():
                    continue
                    
                try:
                    embeddings_file = package_dir / "embeddings.npz"
                    if embeddings_file.exists():
                        data = np.load(embeddings_file)
                        # Package embeddings have shape (1, embedding_dim)
                        package_embedding = data['embeddings'][0]
                        package_name = package_dir.name
                        package_embeddings[package_name] = package_embedding
                except Exception as e:
                    logger.warning(f"Failed to load package embedding for {package_dir.name}: {e}")
        
        logger.info(f"Loaded {len(package_embeddings)} pre-computed package embeddings")
        
        # Concatenate all embeddings
        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)
        else:
            embeddings_array = np.empty((0, 1024), dtype=np.float32)
            
        return embeddings_array, all_metadata, package_embeddings


class SemanticSearchEngine:
    """Main search engine for finding similar OCaml modules."""
    
    def __init__(self, module_embeddings_dir: Path, package_embeddings_dir: Path = None,
                 api_url: str = 'http://localhost:8080', 
                 use_popularity: bool = True, package_descriptions_dir: Path = None):
        """Initialize the semantic search engine."""
        self.query_embedder = QueryEmbedder(api_url)
        self.embedding_loader = PackageEmbeddingLoader(module_embeddings_dir, package_embeddings_dir)
        
        # Load all embeddings
        logger.info("Loading all embeddings into memory...")
        start_time = time.time()
        self.embeddings, self.metadata, self.package_embeddings = self.embedding_loader.load_all_embeddings()
        logger.info(f"Loaded {len(self.embeddings)} embeddings in {time.time() - start_time:.2f}s")
        
        # Initialize popularity scorer if requested
        self.popularity_scorer = None
        if use_popularity:
            self.popularity_scorer = PopularityScorer()
            
        # Load package descriptions for display
        self.package_descriptions = {}
        if package_descriptions_dir is None:
            package_descriptions_dir = Path("package-descriptions")
        self.load_package_descriptions(package_descriptions_dir)
        
        logger.info(f"Search engine ready with {len(self.embeddings)} modules")
        
    def load_package_descriptions(self, descriptions_dir: Path) -> None:
        """Load package descriptions for boosting module rankings."""
        if not descriptions_dir.exists():
            logger.warning(f"Package descriptions directory not found: {descriptions_dir}")
            return
            
        loaded_count = 0
        for json_file in descriptions_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.package_descriptions[data['package']] = data['description']
                    loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load package description {json_file}: {e}")
                
        logger.info(f"Loaded {loaded_count} package descriptions")
    
    def calculate_package_boost(self, query_embedding: np.ndarray, package_names: List[str]) -> Dict[str, float]:
        """Calculate package description similarity boost for modules using pre-computed embeddings."""
        if not self.package_embeddings:
            return {}
        
        # Only calculate for unique packages that have embeddings
        unique_packages = set(package_names)
        
        package_boosts = {}
        boost_count = 0
        
        for package in unique_packages:
            if package in self.package_embeddings:
                # Use pre-computed package embedding
                package_embedding = self.package_embeddings[package]
                
                # Calculate similarity
                similarity = float(np.dot(query_embedding, package_embedding))
                package_boosts[package] = similarity
                boost_count += 1
            else:
                package_boosts[package] = 0.0
        
        logger.info(f"Calculated package boosts for {boost_count} packages (using pre-computed embeddings)")
        return package_boosts
        
    def search(self, query: str, top_k: int = 5, popularity_weight: float = 0.3, 
              package_boost_weight: float = 0.2) -> List[Dict]:
        """
        Search for modules most similar to the given query.
        
        Args:
            query: User's search query
            top_k: Number of results to return
            popularity_weight: Weight for popularity scoring (0-1)
            package_boost_weight: Weight for package description similarity (0-1)
            
        Returns:
            List of search results with module info and scores
        """
        logger.info(f"Searching for: '{query}'")
        
        # Get query embedding
        query_embedding = self.query_embedder.embed_query(query)
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get popularity scores if available
        if self.popularity_scorer:
            popularity_scores = np.array([
                self.popularity_scorer.get_popularity_score(m['module_path']) 
                for m in self.metadata
            ])
        else:
            popularity_scores = np.zeros(len(self.metadata))
            
        # Get package boost scores
        if self.package_embeddings and package_boost_weight > 0:
            # Get package names for all modules
            package_names = [self.metadata[i]['package'] for i in range(len(self.metadata))]
            
            # Calculate package boosts efficiently
            package_boosts = self.calculate_package_boost(query_embedding, package_names)
            
            # Create boost array for all modules
            boost_scores = np.array([package_boosts.get(pkg, 0.0) for pkg in package_names])
        else:
            boost_scores = np.zeros(len(self.metadata))
        
        # Combine scores
        # Normalize weights
        total_weight = 1.0 + popularity_weight + package_boost_weight
        semantic_weight = 1.0 / total_weight
        popularity_weight = popularity_weight / total_weight
        package_boost_weight = package_boost_weight / total_weight
        
        combined_scores = (
            semantic_weight * similarities + 
            popularity_weight * popularity_scores +
            package_boost_weight * boost_scores
        )
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            module_info = self.metadata[idx]
            package_name = module_info['package']
            
            result = {
                'package': package_name,
                'module_path': module_info['module_path'],
                'description': module_info['description'],
                'similarity_score': float(similarities[idx]),
                'popularity_score': float(popularity_scores[idx]),
                'package_boost_score': float(boost_scores[idx]),
                'combined_score': float(combined_scores[idx]),
                'library': module_info.get('library', ''),
            }
            
            # Add package description if available
            if package_name in self.package_descriptions:
                result['package_description'] = self.package_descriptions[package_name]
                
            results.append(result)
            
        return results


def format_results(results: List[Dict], format_type: str = 'text') -> str:
    """Format search results for display."""
    if format_type == 'json':
        return json.dumps(results, indent=2)
    
    # Text format
    lines = []
    for i, result in enumerate(results, 1):
        lines.append(f"\n{i}. {result['package']}.{result['module_path']}")
        if result.get('library'):
            lines.append(f"   Library: {result['library']}")
        lines.append(f"   Similarity: {result['similarity_score']:.3f}")
        if result['popularity_score'] > 0:
            lines.append(f"   Popularity: {result['popularity_score']:.3f}")
        if result['package_boost_score'] > 0:
            lines.append(f"   Package boost: {result['package_boost_score']:.3f}")
        lines.append(f"   Combined: {result['combined_score']:.3f}")
        lines.append(f"   Description: {result['description']}")
        
        # Show package description if available
        if 'package_description' in result:
            lines.append(f"   Package: {result['package_description']}")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Semantic search for OCaml modules")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--module-embeddings-dir", type=Path, default=Path("package-embeddings"),
                       help="Directory containing module embeddings")
    parser.add_argument("--package-embeddings-dir", type=Path, default=Path("package-description-embeddings"),
                       help="Directory containing package description embeddings")
    parser.add_argument("--api-url", default="http://localhost:8080",
                       help="llama-server API URL")
    parser.add_argument("--format", choices=['text', 'json'], default='text',
                       help="Output format")
    parser.add_argument("--no-popularity", action="store_true",
                       help="Disable popularity scoring")
    parser.add_argument("--popularity-weight", type=float, default=0.3,
                       help="Weight for popularity scoring (0-1)")
    parser.add_argument("--package-boost-weight", type=float, default=0.2,
                       help="Weight for package description similarity (0-1)")
    parser.add_argument("--package-descriptions-dir", type=Path, 
                       help="Directory containing package descriptions")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            args.module_embeddings_dir,
            args.package_embeddings_dir,
            args.api_url,
            use_popularity=not args.no_popularity,
            package_descriptions_dir=args.package_descriptions_dir
        )
        
        # Perform search
        results = search_engine.search(
            args.query, 
            args.top_k,
            popularity_weight=args.popularity_weight,
            package_boost_weight=args.package_boost_weight
        )
        
        # Format and print results
        formatted = format_results(results, args.format)
        print(formatted)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())