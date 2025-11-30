"""
Task 2 Fusion Module (LLM-based): Connect LLM knowledge with QA insights
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class FusionModuleLLM:
    """Fuse LLM's internal knowledge with QA insights"""
    
    def __init__(self, llm_client):
        """Initialize with LLM client"""
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def fuse_knowledge_and_data(
        self,
        cluster_action: str,
        topics: List[str],
        concepts: Dict[str, List[str]],
        qa_pairs: List[Dict]
    ) -> str:
        """
        Use LLM to fuse its knowledge about topics with actual QA data.
        
        Args:
            cluster_action: Action label
            topics: Extracted topics FROM QA PAIRS (more reliable)
            concepts: LLM-selected concepts
            qa_pairs: Real QA data
        
        Returns:
            Fused understanding string
        """
        
        topics_str = ', '.join(topics[:4])
        
        # Organize concepts
        causal_str = ', '.join(concepts.get('causal', [])[:2])
        impact_str = ', '.join(concepts.get('impact', [])[:2])
        pattern_str = ', '.join(concepts.get('pattern', [])[:2])
        
        qa_summary = self._summarize_qas(qa_pairs)
        
        prompt = f"""You are performing deep causal reasoning about customer service workflows.

Cluster: {cluster_action}
Topics: {topics_str}

LLM Concepts:
- Root Causes: {causal_str}
- Impacts: {impact_str}
- Patterns: {pattern_str}

Q&A Evidence:
{qa_summary}

TASK:
1. Produce a 3-step CAUSAL CHAIN explaining how the issue evolves.
   Use format: A → B → C → Outcome

   This chain must incorporate:
   - Upstream technical or operational failures
   - Hidden dependencies (billing backend, payment gateway delays)
   - Customer actions or misunderstandings

2. Produce 3 INSIGHTS that combine:
   - Your domain knowledge (telecom/SaaS/billing)
   - The Q&A evidence
   - The concepts listed above

3. Produce 2 INVESTIGATION DIRECTIONS:
   - What analysts should look for in transcript data
   - Which metrics/features should be examined

Return sections as:
CAUSAL_CHAIN:
- A → B → C → Outcome

INSIGHTS:
- ...
- ...
- ...

INVESTIGATION:
- ...
- ...
"""
        
        response = self.llm_client.generate_text(
            prompt,
            max_tokens=400,
            temperature=0.2
        )
        
        self.logger.info("Fused knowledge with QA data")
        
        return response
    
    def _summarize_qas(self, qa_pairs: List[Dict]) -> str:
        """Summarize QA pairs for prompt"""
        
        summary = ""
        for i, qa in enumerate(qa_pairs[:3], 1):
            q = qa.get('query', '')
            a = qa.get('response', '')
            if len(a) > 150:
                a = a[:150] + "..."
            summary += f"Q{i}: {q}\nA{i}: {a}\n\n"
        
        return summary if summary else "No QA data available"
