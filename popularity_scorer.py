"""
Popularity scorer for OCaml modules based on occurrence data.

Provides functionality to load popularity data from occurrences.txt
and compute popularity-adjusted scores for search results.
"""

import math
from typing import Dict, Optional, Tuple, List
from pathlib import Path


class PopularityScorer:
    """Handles popularity scoring based on module occurrence data."""
    
    def __init__(self, occurrences_file: Path = Path("occurrences.txt")):
        """Initialize with occurrence data from file."""
        self.popularity_scores: Dict[str, int] = {}
        self.max_score = 0
        self.load_occurrences(occurrences_file)
    
    def load_occurrences(self, occurrences_file: Path) -> None:
        """Load and parse occurrences from file."""
        if not occurrences_file.exists():
            print(f"Warning: {occurrences_file} not found. Popularity scoring disabled.")
            return
            
        with open(occurrences_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                # Split on last space to handle module names with spaces
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    module_name = parts[0]
                    try:
                        count = int(parts[1])
                        self.popularity_scores[module_name] = count
                        self.max_score = max(self.max_score, count)
                    except ValueError:
                        print(f"Warning: Invalid count in line: {line}")
                        
        print(f"Loaded popularity scores for {len(self.popularity_scores)} modules")
        print(f"Max popularity score: {self.max_score}")
    
    def get_popularity_score(self, module_path: str) -> float:
        """
        Get normalized popularity score for a module path using logarithmic scaling.
        
        Returns a value between 0.0 and 1.0.
        Module paths can be in format: package.Module.Submodule
        We check for exact matches and also parent modules.
        """
        if not self.popularity_scores or self.max_score == 0:
            return 0.0
            
        # Try exact match only
        if module_path in self.popularity_scores:
            raw_score = self.popularity_scores[module_path]
            log_score = math.log(raw_score + 1) / math.log(self.max_score + 1)
            return log_score
                    
        return 0.0
    
    def compute_combined_score(self, relevance_score: float, module_path: str,
                             popularity_weight: float = 0.3) -> float:
        """
        Combine relevance score with popularity score.
        
        Args:
            relevance_score: Original relevance score (e.g., cosine similarity or BM25)
            module_path: Full module path
            popularity_weight: Weight for popularity (0.0 = pure relevance, 1.0 = pure popularity)
            
        Returns:
            Combined score
        """
        popularity = self.get_popularity_score(module_path)
        
        # Weighted combination (popularity is already log-scaled)
        combined = (1 - popularity_weight) * relevance_score + popularity_weight * popularity
        
        return combined
    
    def rerank_results(self, results: List[Tuple], popularity_weight: float = 0.3) -> List[Tuple]:
        """
        Rerank search results based on combined relevance and popularity scores.
        
        Args:
            results: List of tuples (package, module_path, description, score)
            popularity_weight: Weight for popularity in final ranking
            
        Returns:
            Reranked results with updated scores
        """
        if not self.popularity_scores:
            return results
            
        # Compute combined scores
        reranked = []
        for result in results:
            if len(result) >= 4:
                package, module_path, description, score = result[:4]
                
                # Extract module name from full path
                # Format is typically: package.Module.Submodule
                full_path = f"{package}.{module_path}" if package not in module_path else module_path
                
                combined_score = self.compute_combined_score(score, full_path, popularity_weight)
                
                # Preserve original format but with new score
                new_result = (package, module_path, description, combined_score)
                if len(result) > 4:
                    new_result = new_result + result[4:]
                    
                reranked.append(new_result)
            else:
                # Preserve results with unexpected format
                reranked.append(result)
                
        # Sort by new combined scores
        reranked.sort(key=lambda x: x[3] if len(x) >= 4 else 0, reverse=True)
        
        return reranked