from typing import List, Dict, Any
import random
from .llm_client import LLMClient
from .config import Config

class ClusterQueryGenerator:
    """Generate diverse queries from clusters"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or LLMClient()
        self.templates = self._build_templates()
    
    def generate_queries(
        self, 
        cluster_chars: Dict,
        max_queries: int = 50
    ) -> List[str]:
        """
        Generate diverse queries from clusters
        
        Returns: List of query strings
        """
        print(f"Generating queries from {len(cluster_chars)} clusters...")
        
        # 1. Generate seeds
        seeds = self._generate_seeds(cluster_chars)
        print(f"Generated {len(seeds)} seed queries.")
        
        # 2. Expand with LLM
        all_queries = self._expand_with_llm(seeds)
        print(f"Expanded to {len(all_queries)} queries.")
        
        # 3. Filter redundancy
        final_queries = self._enforce_diversity(all_queries)
        print(f"Filtered to {len(final_queries)} diverse queries.")
        
        return final_queries[:max_queries]

    def _build_templates(self) -> Dict[str, str]:
        """Build query templates based on enabled types."""
        templates = {}
        if "pattern_discovery" in Config.QUERY_TEMPLATES_ENABLED:
            templates["pattern_discovery"] = "Why do customers frequently {action} when discussing {theme}?"
        if "outcome_analysis" in Config.QUERY_TEMPLATES_ENABLED:
            templates["outcome_analysis"] = "How does {action} impact the resolution of {theme} issues?"
        if "speaker_dynamics" in Config.QUERY_TEMPLATES_ENABLED:
            templates["speaker_dynamics"] = "What is the typical agent response when a customer {action} regarding {theme}?"
        if "comparative" in Config.QUERY_TEMPLATES_ENABLED:
            templates["comparative"] = "Compare the outcomes of {action} versus other actions in {theme} contexts."
        return templates

    def _generate_seeds(self, cluster_chars: Dict) -> List[str]:
        """Generate parametrized seed queries."""
        seeds = []
        for cid, data in cluster_chars.items():
            action = data['action_label']
            # Use top key phrase as theme, or fallback to action
            theme = data['key_phrases'][0] if data['key_phrases'] else action
            
            for tmpl_name, tmpl_str in self.templates.items():
                query = tmpl_str.format(action=action, theme=theme)
                seeds.append(query)
        return seeds

    def _expand_with_llm(self, seeds: List[str]) -> List[str]:
        """Expand seeds into multiple variations using LLM."""
        expanded_queries = []
        # Process in batches to save calls, or one by one. 
        # For simplicity and speed in this refactor, we'll do a simple prompt for each seed 
        # or a batch prompt. Let's do a batch prompt for efficiency if list is long, 
        # but here we'll iterate.
        
        # Limit seeds to avoid excessive API calls during testing
        max_seeds = 10 
        process_seeds = seeds[:max_seeds]
        
        for seed in process_seeds:
            prompt = (
                f"Generate {Config.N_QUERIES_PER_SEED} diverse, specific causal queries "
                f"based on this seed: '{seed}'. "
                f"Focus on business impact and root cause. "
                f"Return only the queries, one per line."
            )
            
            try:
                response = self.llm_client.generate_text(prompt)
                if response:
                    queries = [q.strip() for q in response.split('\n') if q.strip()]
                    expanded_queries.extend(queries)
                else:
                    # Fallback if response is None (e.g. disabled LLM)
                    print(f"⚠️  LLM returned None for seed '{seed}'. Using seed as fallback.")
                    expanded_queries.append(seed)
            except Exception as e:
                print(f"Error expanding seed '{seed}': {e}")
                # Fallback: keep the seed itself
                expanded_queries.append(seed)
                
        return expanded_queries

    def _enforce_diversity(self, queries: List[str]) -> List[str]:
        """
        Remove semantically similar queries.
        Simple implementation: deduplicate by string exact match first, 
        then could use embeddings if available. 
        For this MVP, we'll use string deduplication and length filtering.
        """
        unique_queries = list(set(queries))
        # Filter too short/long
        filtered = [q for q in unique_queries if 10 < len(q.split()) < 30]
        return filtered if filtered else unique_queries
