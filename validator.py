"""
Query validator using LLM-as-a-judge with 3-point rubric.

Scores queries on:
- Causal Depth (1-5): Root cause identification
- Clarity (1-5): Unambiguous phrasing
- Business Value (1-5): Actionable for RCA

Aggregate score > 3.5 = ACCEPT
"""

import logging
from typing import Optional

from .schemas import QueryValidationScore
from .llm_client import LLMClient
from .config import Config

logger = logging.getLogger(__name__)


class QueryValidator:
    """
    Validate queries using LLM judge with 3-point rubric.
    
    Uses cheaper model (Claude Haiku) for cost-efficient scoring.
    Implements consistent rubric application per design doc.
    """
    
    def __init__(self, judge_client: Optional[LLMClient] = None):
        """
        Initialize validator.
        
        Args:
            judge_client: LLM client for judging (uses config default if None)
        """
        if judge_client is None:
            # Create judge client with cheaper model
            judge_config = Config.get_judge_config()
            self.judge_client = LLMClient(
                model=judge_config["model"],
                temperature=judge_config["temperature"],
                max_tokens=judge_config["max_tokens"]
            )
        else:
            self.judge_client = judge_client
        
        self.acceptance_threshold = Config.VALIDATION_ACCEPTANCE_THRESHOLD
    
    def validate(
        self,
        query: str,
        event_type: str,
        transcript_excerpt: str
    ) -> QueryValidationScore:
        """
        Validate a query using 3-point rubric.
        
        Args:
            query: Generated causal query
            event_type: Business event type
            transcript_excerpt: Context from transcript (first 500 chars)
        
        Returns:
            QueryValidationScore with scores and decision
        """
        # Build validation prompt
        prompt = self._build_validation_prompt(
            query=query,
            event_type=event_type,
            context=transcript_excerpt[:500]
        )
        
        # Offline fast-path
        if Config.DISABLE_LLM:
            # Conservative but accepting defaults to enable CSV creation offline
            aggregate = round((4 + 4 + 3) / 3.0, 2)
            return QueryValidationScore(
                causal_depth=4,
                clarity=4,
                business_value=3,
                aggregate_score=aggregate,
                decision="ACCEPT" if aggregate > self.acceptance_threshold else "REJECT",
                reasoning="Offline validation: heuristic acceptance in LLM-disabled mode"
            )

        # Get scores from judge LLM
        result = self.judge_client.generate_structured(
            prompt=prompt,
            response_model=QueryValidationScore,
            system_prompt="You are an expert quality evaluator for causal queries."
        )
        
        if result is None:
            # Fallback: reject on failure
            logger.warning("Validation LLM failed, defaulting to REJECT")
            return QueryValidationScore(
                causal_depth=1,
                clarity=1,
                business_value=1,
                aggregate_score=1.0,
                decision="REJECT",
                reasoning="Validation error: LLM judge failed"
            )
        
        # Compute aggregate score
        aggregate = (
            result.causal_depth + result.clarity + result.business_value
        ) / 3.0
        
        result.aggregate_score = round(aggregate, 2)
        
        # Make decision
        result.decision = "ACCEPT" if aggregate > self.acceptance_threshold else "REJECT"
        
        logger.debug(
            f"Validation: {result.decision} "
            f"(depth={result.causal_depth}, clarity={result.clarity}, "
            f"value={result.business_value}, agg={result.aggregate_score})"
        )
        
        return result
    
    def _build_validation_prompt(
        self,
        query: str,
        event_type: str,
        context: str
    ) -> str:
        """
        Build validation prompt with rubric.
        
        Args:
            query: Query to validate
            event_type: Business event type
            context: Transcript context
        
        Returns:
            Formatted prompt
        """
        prompt = f"""Rate this causal query on 3 dimensions (1-5 each).

QUERY: {query}
EVENT_TYPE: {event_type}
CONTEXT: {context}...

RUBRIC:

CAUSAL_DEPTH (1-5):
5 = Identifies root cause of {event_type} (explains WHY event happened)
4 = Direct agent action → outcome causal link
3 = Correlation or pattern observation
2 = Fact retrieval (who, what, when)
1 = Not causal / malformed question

CLARITY (1-5):
5 = Unambiguous single question, proper grammar, clear intent
4 = Clear question, minor grammar issues
3 = Mostly clear, some ambiguity in phrasing
2 = Ambiguous, multiple possible interpretations
1 = Unintelligible or confusing

BUSINESS_VALUE (1-5):
5 = Actionable for agent coaching or compliance RCA
4 = Useful for {event_type} root cause analysis
3 = Useful for general analytics
2 = Interesting but limited operational use
1 = Not useful for business operations

INSTRUCTIONS:
- Score each dimension independently
- Provide brief reasoning (1-2 sentences)
- Be strict but fair
- Focus on whether the query is answerable from the transcript
- Penalize hallucinations or unanswerable questions

Return your scores and reasoning following the schema."""

        return prompt
    
    def validate_batch(
        self,
        queries: list,
        event_types: list,
        contexts: list
    ) -> list:
        """
        Validate multiple queries.
        
        Args:
            queries: List of query strings
            event_types: Corresponding event types
            contexts: Corresponding transcript excerpts
        
        Returns:
            List of QueryValidationScore objects
        """
        results = []
        
        for query, event_type, context in zip(queries, event_types, contexts):
            result = self.validate(
                query=query,
                event_type=event_type,
                transcript_excerpt=context
            )
            results.append(result)
        
        return results
    
    def get_acceptance_rate(self, validation_results: list) -> float:
        """
        Calculate acceptance rate from validation results.
        
        Args:
            validation_results: List of QueryValidationScore objects
        
        Returns:
            Acceptance rate (0.0 to 1.0)
        """
        if not validation_results:
            return 0.0
        
        accepted = sum(1 for r in validation_results if r.decision == "ACCEPT")
        return accepted / len(validation_results)
    
    def get_average_scores(self, validation_results: list) -> dict:
        """
        Calculate average scores across all dimensions.
        
        Args:
            validation_results: List of QueryValidationScore objects
        
        Returns:
            Dict with average scores
        """
        if not validation_results:
            return {
                "causal_depth": 0.0,
                "clarity": 0.0,
                "business_value": 0.0,
                "aggregate": 0.0
            }
        
        n = len(validation_results)
        
        return {
            "causal_depth": sum(r.causal_depth for r in validation_results) / n,
            "clarity": sum(r.clarity for r in validation_results) / n,
            "business_value": sum(r.business_value for r in validation_results) / n,
            "aggregate": sum(r.aggregate_score for r in validation_results) / n
        }
