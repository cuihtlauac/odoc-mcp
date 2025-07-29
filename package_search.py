#!/usr/bin/env python3
"""
Package-level semantic search for OCaml packages.

This script searches across package descriptions (not individual modules)
to find packages that match a given query.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Workaround for transformers 4.52.4 bug with ALL_PARALLEL_STYLES
import transformers.modeling_utils
if transformers.modeling_utils.ALL_PARALLEL_STYLES is None:
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
        # Task description for OCaml package search
        task = 'Given a description of desired functionality, retrieve relevant OCaml packages that provide that functionality'
        
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

class PackageSearch:
    """Package-level semantic search engine."""
    
    def __init__(self, embeddings_dir: Path, model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
        """Initialize the package search engine."""
        self.embeddings_dir = embeddings_dir
        self.query_embedder = QueryEmbedder(model_name)
        
        # Load embeddings and metadata
        self.embeddings, self.metadata = self.load_embeddings()
        
        logger.info(f"Package search ready with {len(self.embeddings)} packages")
    
    def load_embeddings(self) -> tuple:
        """Load package embeddings and metadata from per-package structure."""
        packages_dir = self.embeddings_dir / "packages"
        metadata_file = self.embeddings_dir / "metadata.json"
        
        if not packages_dir.exists():
            raise FileNotFoundError(f"Packages directory not found: {packages_dir}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load global metadata
        with open(metadata_file, 'r') as f:
            global_metadata = json.load(f)
        
        # Load embeddings from each package
        embeddings_list = []
        package_names = []
        
        for package_name in global_metadata['packages']:
            package_dir = packages_dir / package_name
            embeddings_file = package_dir / "embeddings.npz"
            
            if embeddings_file.exists():
                embeddings_data = np.load(embeddings_file)
                embedding = embeddings_data['embeddings'][0]  # Remove batch dimension
                embeddings_list.append(embedding)
                package_names.append(package_name)
        
        # Convert to numpy array
        embeddings = np.array(embeddings_list)
        
        # Create metadata structure
        metadata = {
            'packages': package_names,
            'total_packages': len(package_names),
            'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0
        }
        
        logger.info(f"Loaded {len(embeddings)} package embeddings")
        
        return embeddings, metadata
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for packages most similar to the given query.
        
        Args:
            query: User's search query
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing package information and similarity scores
        """
        logger.info(f"Searching for: '{query}'")
        start_time = time.time()
        
        # Embed the query
        query_embedding = self.query_embedder.embed_query(query)
        
        # Calculate similarities
        similarities = self.calculate_similarities(query_embedding)
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Descending order
        
        # Format results
        results = []
        for i, idx in enumerate(top_indices):
            result = {
                'rank': i + 1,
                'similarity_score': float(similarities[idx]),
                'package': self.metadata['packages'][idx]
            }
            results.append(result)
            
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f}s")
        
        return results
    
    def calculate_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between query and all package embeddings."""
        # Embeddings are already normalized, so dot product gives cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)
        return similarities

def load_package_descriptions(descriptions_dir: Path) -> Dict[str, Dict]:
    """Load package descriptions for displaying in results."""
    descriptions = {}
    
    for json_file in descriptions_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'package' in data:
                descriptions[data['package']] = data
                
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    return descriptions

def format_results(results: List[Dict], descriptions: Dict[str, Dict], format_type: str = 'text') -> str:
    """Format search results for display."""
    if format_type == 'json':
        # Add descriptions to results for JSON output
        enhanced_results = []
        for result in results:
            enhanced_result = result.copy()
            if result['package'] in descriptions:
                enhanced_result.update(descriptions[result['package']])
            enhanced_results.append(enhanced_result)
        return json.dumps(enhanced_results, indent=2)
    
    elif format_type == 'text':
        output = []
        output.append(f"Top {len(results)} most relevant OCaml packages:\n")
        
        for result in results:
            output.append(f"#{result['rank']} - {result['package']}")
            output.append(f"  Similarity: {result['similarity_score']:.4f}")
            
            # Add package description if available
            if result['package'] in descriptions:
                pkg_info = descriptions[result['package']]
                if 'version' in pkg_info:
                    output.append(f"  Version: {pkg_info['version']}")
                if 'description' in pkg_info:
                    output.append(f"  Description: {pkg_info['description']}")
            
            output.append("")
            
        return '\n'.join(output)
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Semantic search for OCaml packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python package_search.py "HTTP client library"
  python package_search.py "JSON parsing and serialization" --top-k 10
  python package_search.py "machine learning" --format json
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
        '--descriptions-dir',
        type=Path,
        default=Path('package-descriptions'),
        help='Directory containing package descriptions (default: package-descriptions)'
    )
    
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-Embedding-0.6B',
        help='Embedding model to use (default: Qwen/Qwen3-Embedding-0.6B)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top results to return (default: 10)'
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
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize search engine
        search_engine = PackageSearch(args.embeddings_dir, args.model)
        
        # Load package descriptions for display
        descriptions = load_package_descriptions(args.descriptions_dir)
        
        # Perform search
        results = search_engine.search(args.query, args.top_k)
        
        # Format and display results
        formatted_output = format_results(results, descriptions, args.format)
        print(formatted_output)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())