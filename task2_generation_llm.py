"""
Task 2 Generation Module: Generate follow-up questions using fused understanding
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)


class GenerationModuleLLM:
    """Generate follow-up questions using fused knowledge"""
    
    def __init__(self, llm_client):
        """Initialize with LLM client"""
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def generate_followups(
        self,
        cluster_action: str,
        topics: List[str],
        concepts: Dict[str, List[str]],
        fused_understanding: str,
        qa_pairs: List[Dict],
        max_followups: int = 3
    ) -> List[Dict]:
        """
        Generate diverse follow-up questions.
        
        Args:
            cluster_action: Action label
            topics: Extracted topics FROM QA PAIRS
            concepts: Selected concepts
            fused_understanding: Output from Fusion module
            qa_pairs: Real QA data
            max_followups: Max questions
        
        Returns:
            List of dicts with questions
        """
        
        topics_str = ', '.join(topics[:5])
        qa_context = self._format_qas(qa_pairs)
        
        prompt = f"""You are generating DEEP follow-up questions for analysts.

Your outputs MUST:
- require domain knowledge to answer
- reference hidden mechanisms or failure modes
- be grounded in Q&A evidence OR concepts from fusion
- drive investigation, not just ask obvious questions

Context:
Cluster: {cluster_action}
Topics: {topics_str}

Fused Knowledge:
{fused_understanding}

Q&A Context:
{qa_context}

Generate 3 questions:

QUESTION 1 — ROOT CAUSE DEPTH:
Ask about upstream/systemic causes that require analysis of backend systems,
reconciliation processes, or customer behavior patterns.
This must involve multi-hop reasoning.

QUESTION 2 — QUANT/IMPACT ANALYSIS:
Ask a question that requires measuring a business/operational metric
(e.g., churn, SLA breach, refund processing time variations,
frequency of backend delays).
It must be answerable by data analysis.

QUESTION 3 — PATTERN / SCENARIO DISCOVERY:
Ask about conditional situations:
“When X and Y happen together, does Z increase?”
Should require correlational / pattern mining logic.

FORMATTING:
Return them as:

QUESTION_1:
...

QUESTION_2:
...

QUESTION_3:
...
"""
        
        response = self.llm_client.generate_text(
            prompt,
            max_tokens=400,
            temperature=0.6
        )
        
        self.logger.info(f"Raw LLM Response:\n{response}")
        
        # Parse questions
        followups = self._parse_questions(response)
        
        self.logger.info(f"Generated {len(followups)} follow-up questions")
        
        return followups[:max_followups]
    
    def _format_qas(self, qa_pairs: List[Dict]) -> str:
        """Format QA pairs for prompt"""
        
        qa_text = ""
        for i, qa in enumerate(qa_pairs[:2], 1):
            q = qa.get('query', '')
            a = qa.get('response', '')
            if len(a) > 100:
                a = a[:100] + "..."
            qa_text += f"Q{i}: {q}\nA{i}: {a}\n\n"
        
        return qa_text if qa_text else "No additional context"
    
    def _parse_questions(self, response: str) -> List[Dict]:
        """Extract follow-up questions from response"""
        
        questions = []
        
        # Find questions after QUESTION_N: markers
        pattern = r'QUESTION_\d+:\s*(.+?)(?=QUESTION_\d+:|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        categories = ['causal', 'impact', 'pattern']
        
        for i, match in enumerate(matches[:3]):
            lines = match.strip().split('\n')
            
            # Find first non-empty line that looks like a question
            for line in lines:
                q = line.strip()
                
                # Validate
                if not q or len(q) < 10:
                    continue
                if '?' not in q:
                    continue
                if len(q.split()) > 75:
                    continue
                
                # Clean up
                q = re.sub(r'^\[.*?\]\s*', '', q)
                q = q.strip()
                
                questions.append({
                    'question': q,
                    'category': categories[i] if i < len(categories) else 'other',
                    'sequence': i + 1
                })
                
                break
        
        return questions[:3]
