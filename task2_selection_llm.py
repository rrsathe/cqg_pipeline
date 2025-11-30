"""
Task 2 Selection Module (LLM-based): Select relevant concepts from LLM knowledge

Input: topics (from QA pairs) + qa_pairs
Process: Ask LLM to identify causes, impacts, patterns
Output: Concepts organized by category
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


class SelectionModuleLLM:
    """Use LLM's internal knowledge to select relevant concepts"""
    
    def __init__(self, llm_client):
        """Initialize with LLM client"""
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def select_relevant_concepts(
        self,
        cluster_action: str,
        topics: List[str],
        qa_pairs: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Use LLM to identify relevant concepts from its knowledge.
        
        Args:
            cluster_action: Action label (e.g., 'refund_request')
            topics: Extracted topics FROM QA PAIRS (now more reliable)
            qa_pairs: QA pairs for this cluster
        
        Returns:
            Dictionary with concepts by category:
            {'causal': [...], 'impact': [...], 'pattern': [...]}
        """
        
        if not topics:
            self.logger.warning("No topics provided")
            return {'causal': [], 'impact': [], 'pattern': []}
        
        # Build prompt
        topics_str = ', '.join(topics[:5])
        qa_summary = self._summarize_qas(qa_pairs)
        
        prompt = f"""You are an expert in telecom billing systems, subscription lifecycle operations, 
customer retention, refund processing, payment gateways, and contact center analytics.

Your job is to analyze this cluster and identify DEEP, NON-OBVIOUS concepts that 
go beyond the Q&A data, using your internal knowledge and reasoning.

Cluster Action: {cluster_action}
Key Topics: {topics_str}

Here is the Q&A evidence:
{qa_summary}

Provide:
1. ROOT CAUSES (3 items):
   - Systemic or upstream process issues
   - Typical industry failure modes
   - Operational bottlenecks not mentioned explicitly in the Q&A

2. IMPACTS (3 items):
   - Business or operational consequences
   - Customer experience metrics
   - Revenue or SLA implications

3. PATTERNS / CONDITIONS (3 items):
   - Common scenarios when the issue occurs
   - Latent conditions (e.g., reconciliation periods, API failures)
   - Related behaviors seen in telecom/customer support

GROUNDING RULES:
- Use world knowledge of telecom and SaaS systems.
- Tie each concept back to the Q&A *implicitly or explicitly*.
- Do NOT hallucinate specific numbers not in the Q&A.

Format EXACTLY:

ROOT_CAUSES:
- ...
- ...
- ...

IMPACTS:
- ...
- ...
- ...

PATTERNS:
- ...
- ...
- ...
"""
        
        # Call LLM
        response = self.llm_client.generate_text(
            prompt, 
            max_tokens=400, 
            temperature=0.2
        )
        
        # Parse concepts
        concepts = self._parse_concepts(response)
        
        self.logger.info(
            f"Selected {sum(len(v) for v in concepts.values())} concepts"
        )
        
        return concepts
    
    def _summarize_qas(self, qa_pairs: List[Dict]) -> str:
        """Summarize QA pairs for prompt"""
        
        summary = ""
        for i, qa in enumerate(qa_pairs[:3], 1):
            q = qa.get('query', '')[:60]
            a = qa.get('response', '')[:120]
            summary += f"Q{i}: {q}\nA{i}: {a}...\n\n"
        
        return summary if summary else "No QA data available"
    
    def _parse_concepts(self, response: str) -> Dict[str, List[str]]:
        """Parse LLM response to extract concepts by category"""
        
        concepts = {'causal': [], 'impact': [], 'pattern': []}
        
        lines = response.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            
            # Detect category headers
            if 'ROOT_CAUSE' in line.upper() or 'CAUSAL' in line.upper():
                current_category = 'causal'
                continue
            elif 'IMPACT' in line.upper():
                current_category = 'impact'
                continue
            elif 'SCENARIO' in line.upper() or 'PATTERN' in line.upper():
                current_category = 'pattern'
                continue
            
            # Extract concepts
            if line and (line.startswith('-') or line.startswith('•')):
                concept = line[1:].strip()
                if concept and len(concept) > 2 and current_category:
                    concept = concept.rstrip('.,;:')
                    concepts[current_category].append(concept)
        
        return concepts
