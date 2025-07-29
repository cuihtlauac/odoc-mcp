#!/usr/bin/env python3
"""
Semantic Search for OCaml Modules

This script embeds user queries using Qwen3-Embedding-0.6B and finds the most
semantically similar OCaml modules from the package embeddings dataset.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from popularity_scorer import PopularityScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Workaround for transformers 4.52.4 bug with ALL_PARALLEL_STYLES
import transformers.modeling_utils
if transformers.modeling_utils.ALL_PARALLEL_STYLES is None:
    # Define expected parallel styles from transformers library
    transformers.modeling_utils.ALL_PARALLEL_STYLES = ["colwise", "rowwise"]
    logger.warning("Patched ALL_PARALLEL_STYLES bug in transformers library")


class QueryEmbedder:
    """Handles embedding of user queries using Qwen3-Embedding-0.6B model."""
    
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
        """Initialize the query embedder with the specified model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model loaded on GPU")
        else:
            logger.info("Model loaded on CPU")
            
        self.max_length = 8192
        
    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last token from the hidden states."""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format the query with task instruction."""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query and return the normalized embedding vector.
        
        Args:
            query: The user's search query
            
        Returns:
            Normalized embedding vector as numpy array
        """
        # Task description for OCaml module search
        task = 'Given a programming task or functionality description, retrieve relevant OCaml modules that provide that functionality'
        
        # Format query with instruction
        formatted_query = self.get_detailed_instruct(task, query)
        
        # Tokenize
        batch_dict = self.tokenizer(
            [formatted_query],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Move to same device as model
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        # Convert to numpy and return
        return embeddings.cpu().numpy()[0]


class PackageEmbeddingLoader:
    """Handles loading and managing package embeddings from the dataset."""
    
    def __init__(self, embeddings_dir: Path):
        """Initialize the loader with the embeddings directory."""
        self.embeddings_dir = embeddings_dir
        self.packages_dir = embeddings_dir / "packages"
        
        if not self.packages_dir.exists():
            raise FileNotFoundError(f"Embeddings directory not found: {self.packages_dir}")
            
        # Build package index
        self.package_index = self._build_package_index()
        logger.info(f"Found {len(self.package_index)} packages with embeddings")
        
    def _build_package_index(self) -> Dict[str, Dict]:
        """Build an index of available packages and their metadata."""
        index = {}
        
        for package_dir in self.packages_dir.iterdir():
            if not package_dir.is_dir():
                continue
                
            embeddings_file = package_dir / "embeddings.npz"
            metadata_file = package_dir / "metadata.json"
            
            if embeddings_file.exists() and metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    index[package_dir.name] = {
                        'embeddings_path': embeddings_file,
                        'metadata_path': metadata_file,
                        'metadata': metadata,
                        'num_modules': len(metadata.get('modules', []))
                    }
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {package_dir.name}: {e}")
                    
        return index
    
    def load_all_embeddings(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load all embeddings into memory and return consolidated arrays.
        
        Returns:
            Tuple of (embeddings_array, module_metadata_list)
        """
        logger.info("Loading all embeddings into memory...")
        start_time = time.time()
        
        all_embeddings = []
        all_metadata = []
        
        for package_name, package_info in self.package_index.items():
            try:
                # Load embeddings
                embeddings_data = np.load(package_info['embeddings_path'])
                embeddings = embeddings_data['embeddings']
                
                # Add to consolidated arrays
                all_embeddings.append(embeddings)
                
                # Add metadata for each module
                for module_info in package_info['metadata']['modules']:
                    module_metadata = {
                        'package': package_name,
                        'module_path': module_info['module_path'],
                        'description': module_info['description'],
                        'description_length': module_info['description_length'],
                        'index_in_package': module_info['index']
                    }
                    # Add library information if available
                    if 'library' in module_info:
                        module_metadata['library'] = module_info['library']
                    all_metadata.append(module_metadata)
                    
            except Exception as e:
                logger.warning(f"Failed to load embeddings for {package_name}: {e}")
                
        # Concatenate all embeddings
        if all_embeddings:
            consolidated_embeddings = np.vstack(all_embeddings)
            logger.info(f"Loaded {len(consolidated_embeddings)} embeddings in {time.time() - start_time:.2f}s")
            return consolidated_embeddings, all_metadata
        else:
            raise RuntimeError("No embeddings could be loaded")


class SemanticSearch:
    """Main semantic search engine that combines query embedding and similarity search."""
    
    def __init__(self, embeddings_dir: Path, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', 
                 use_popularity: bool = True, package_descriptions_dir: Path = None):
        """Initialize the semantic search engine."""
        self.query_embedder = QueryEmbedder(model_name)
        self.embedding_loader = PackageEmbeddingLoader(embeddings_dir)
        
        # Load all embeddings into memory
        self.embeddings, self.metadata = self.embedding_loader.load_all_embeddings()
        
        # Initialize popularity scorer if enabled
        self.use_popularity = use_popularity
        self.popularity_scorer = PopularityScorer() if use_popularity else None
        
        # Load package descriptions for boosting
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
                
                if 'package' in data and 'description' in data:
                    self.package_descriptions[data['package']] = data['description']
                    loaded_count += 1
                    
            except Exception as e:
                logger.warning(f"Failed to load package description {json_file}: {e}")
                
        logger.info(f"Loaded {loaded_count} package descriptions")
    
    def calculate_package_boost(self, query: str, package_names: List[str]) -> Dict[str, float]:
        """Calculate package description similarity boost for modules."""
        if not self.package_descriptions:
            return {}
            
        # Get query embedding once
        query_embedding = self.query_embedder.embed_query(query)
        
        # Only calculate for unique packages that have descriptions
        unique_packages = set(package_names)
        packages_with_descriptions = [p for p in unique_packages if p in self.package_descriptions]
        
        logger.info(f"Calculating package boosts for {len(packages_with_descriptions)} packages")
        
        package_boosts = {}
        for package in unique_packages:
            if package in self.package_descriptions:
                # Get package description embedding
                package_desc = self.package_descriptions[package]
                package_embedding = self.query_embedder.embed_query(package_desc)
                
                # Calculate similarity
                similarity = float(np.dot(query_embedding, package_embedding))
                package_boosts[package] = similarity
            else:
                package_boosts[package] = 0.0
                
        return package_boosts
        
    def search(self, query: str, top_k: int = 5, popularity_weight: float = 0.3, 
              package_boost_weight: float = 0.2) -> List[Dict]:
        """
        Search for modules most similar to the given query.
        
        Args:
            query: User's search query
            top_k: Number of top results to return
            popularity_weight: Weight for popularity in combined score (0.0-1.0)
            package_boost_weight: Weight for package description boost (0.0-1.0)
            
        Returns:
            List of dictionaries containing module information and similarity scores
        """
        logger.info(f"Searching for: '{query}'")
        start_time = time.time()
        
        # Embed the query
        query_embedding = self.query_embedder.embed_query(query)
        
        # Calculate similarities
        similarities = self._calculate_similarities(query_embedding)
        
        # Apply package boosting if enabled
        if self.package_descriptions and package_boost_weight > 0:
            # Get package names for all modules
            package_names = [self.metadata[i]['package'] for i in range(len(self.metadata))]
            
            # Calculate package boosts
            package_boosts = self.calculate_package_boost(query, package_names)
            
            # Apply package boost to similarities
            for i, package in enumerate(package_names):
                boost = package_boosts.get(package, 0.0)
                similarities[i] = similarities[i] * (1 - package_boost_weight) + boost * package_boost_weight
        
        # Get initial top results (retrieve more for reranking)
        initial_top_k = min(top_k * 3, len(similarities))  # Get 3x results for reranking
        top_indices = np.argsort(similarities)[-initial_top_k:][::-1]  # Descending order
        
        # Format initial results
        results = []
        for idx in top_indices:
            result = {
                'similarity_score': float(similarities[idx]),
                'package': self.metadata[idx]['package'],
                'module_path': self.metadata[idx]['module_path'],
                'description': self.metadata[idx]['description'],
                'description_length': self.metadata[idx]['description_length']
            }
            # Include library information if available
            if 'library' in self.metadata[idx]:
                result['library'] = self.metadata[idx]['library']
            results.append(result)
        
        # Apply popularity-based reranking if enabled
        if self.use_popularity and self.popularity_scorer:
            # Convert to tuple format expected by rerank_results
            result_tuples = [(r['package'], r['module_path'], r['description'], r['similarity_score']) 
                           for r in results]
            
            # Rerank with popularity
            reranked_tuples = self.popularity_scorer.rerank_results(result_tuples, popularity_weight)
            
            # Convert back to dict format and keep only top_k
            final_results = []
            for i, (package, module_path, description, score) in enumerate(reranked_tuples[:top_k]):
                # Get original similarity score for debugging
                original_score = None
                for orig_result in results:
                    if (orig_result['package'] == package and 
                        orig_result['module_path'] == module_path):
                        original_score = orig_result['similarity_score']
                        break
                
                # Get popularity score for this module
                popularity_score = self.popularity_scorer.get_popularity_score(module_path)
                
                # Debug: show what we're looking up
                if popularity_score > 0:
                    logger.info(f"Found popularity for {module_path}: {popularity_score:.4f}")
                else:
                    logger.debug(f"No popularity data for {module_path}")
                
                result = {
                    'rank': i + 1,
                    'similarity_score': score,
                    'original_similarity': original_score,
                    'popularity_score': popularity_score,
                    'package': package,
                    'module_path': module_path,
                    'description': description,
                }
                # Find original metadata to add extra fields
                for meta in self.metadata:
                    if meta['package'] == package and meta['module_path'] == module_path:
                        result['description_length'] = meta['description_length']
                        if 'library' in meta:
                            result['library'] = meta['library']
                        break
                final_results.append(result)
            results = final_results
        else:
            # No popularity scoring, just add ranks to final results
            results = results[:top_k]
            for i, result in enumerate(results):
                result['rank'] = i + 1
            
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f}s")
        
        return results
    
    def _calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between query and all module embeddings."""
        # Embeddings are already normalized, so dot product gives cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        return similarities


def format_results(results: List[Dict], format_type: str = 'text') -> str:
    """Format search results for display."""
    if format_type == 'json':
        return json.dumps(results, indent=2)
    
    elif format_type == 'text':
        output = []
        output.append(f"Top {len(results)} most relevant OCaml modules:\n")
        
        for result in results:
            output.append(f"#{result['rank']} - {result['package']}: {result['module_path']}")
            output.append(f"  Similarity: {result['similarity_score']:.4f}")
            
            # Add popularity debugging info if available
            if 'popularity_score' in result:
                output.append(f"  Original Similarity: {result.get('original_similarity', 'N/A'):.4f}")
                output.append(f"  Popularity Score: {result['popularity_score']:.4f}")
                
            output.append(f"  Description: {result['description']}")
            output.append("")
            
        return '\n'.join(output)
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Semantic search for OCaml modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python semantic_search.py "http server"
  python semantic_search.py "parse JSON data" --top-k 10
  python semantic_search.py "crypto hash function" --format json
        """
    )
    
    parser.add_argument(
        'query',
        help='Search query describing the desired functionality'
    )
    
    parser.add_argument(
        '--embeddings-dir',
        type=Path,
        default=Path('package-embeddings'),
        help='Directory containing package embeddings (default: package-embeddings)'
    )
    
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-Embedding-0.6B',
        help='Embedding model to use (default: Qwen/Qwen3-Embedding-0.6B)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to return (default: 5)'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-popularity',
        action='store_true',
        help='Disable popularity-based ranking'
    )
    
    parser.add_argument(
        '--popularity-weight',
        type=float,
        default=0.3,
        help='Weight for popularity in ranking (0.0-1.0, default: 0.3)'
    )
    
    parser.add_argument(
        '--package-boost-weight',
        type=float,
        default=0.2,
        help='Weight for package description boost (0.0-1.0, default: 0.2)'
    )
    
    parser.add_argument(
        '--package-descriptions-dir',
        type=Path,
        default=Path('package-descriptions'),
        help='Directory containing package descriptions (default: package-descriptions)'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize search engine
        search_engine = SemanticSearch(args.embeddings_dir, args.model, 
                                     use_popularity=not args.no_popularity,
                                     package_descriptions_dir=args.package_descriptions_dir)
        
        # Perform search
        results = search_engine.search(args.query, args.top_k, 
                                     popularity_weight=args.popularity_weight,
                                     package_boost_weight=args.package_boost_weight)
        
        # Format and display results
        formatted_output = format_results(results, args.format)
        print(formatted_output)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())