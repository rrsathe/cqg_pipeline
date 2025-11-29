"""
Pydantic data models for CQG pipeline.

Defines structured schemas for:
- Causal queries
- Validation scores
- Evidence grounding results
- Event detection results
"""

import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class CausalQuerySchema(BaseModel):
    """Schema for a generated causal query with strict anti-trivial validation."""
    
    query: str = Field(
        description="Single question, max 50 words, Why/How format"
    )
    event_type: str = Field(
        description="Classified event type (escalation, churn, refund, compliance, general)"
    )
    causal_trigger: str = Field(
        description="Agent action or event being probed"
    )
    causal_outcome: str = Field(
        description="Business outcome being tested"
    )
    lehnert_category: str = Field(
        description="One of: antecedent, consequent, goal, enablement"
    )
    complexity_level: int = Field(
        ge=2,
        le=5,
        description="Bloom's taxonomy level (2-5)"
    )
    
    @field_validator('query')
    @classmethod
    def validate_causal_query(cls, v: str) -> str:
        """Validate query is truly causal, not trivial/factual."""
        if not v:
            raise ValueError("Query cannot be empty")
        
        # Word count check
        if len(v.split()) > 50:
            raise ValueError(f"Query exceeds 50 words ({len(v.split())} words)")
        
        v_lower = v.lower()
        
        # Must start with Why/How (or What with causal verb)
        if not v_lower.startswith(('why', 'how')):
            if v_lower.startswith('what'):
                causal_verbs = ['cause', 'lead', 'influence', 'result', 'impact', 'trigger', 'motivate']
                if not any(verb in v_lower for verb in causal_verbs):
                    raise ValueError("What-questions must contain causal verbs (cause, lead, influence, etc.)")
            else:
                raise ValueError("Query must start with Why, How, or What (with causal verb)")
        
        # Block trivial factual keywords
        trivial = ['what date', 'what time', 'what hotel', 'what room', 'which agent', 
                   'booking id', 'phone number', 'confirmation number']
        if any(t in v_lower for t in trivial):
            raise ValueError(f"Query contains trivial factual keyword")
        
        # Detect factual patterns
        factual_patterns = [
            r'\bwhat (did|was|is)\b.*\b(mention|say|tell)\b',
            r'\bwho (said|mentioned)\b'
        ]
        if any(re.search(p, v_lower) for p in factual_patterns):
            raise ValueError("Query matches factual pattern (not causal)")
        
        return v
    
    @field_validator('causal_trigger')
    @classmethod
    def validate_trigger_specificity(cls, v: str) -> str:
        """Ensure trigger is specific, not generic."""
        if not v or v.lower() in ['the agent', 'agent action', 'customer']:
            raise ValueError("causal_trigger must be specific, not generic")
        return v
    
    @field_validator('causal_outcome')
    @classmethod
    def validate_outcome_specificity(cls, v: str) -> str:
        """Ensure outcome is specific, not generic."""
        if not v or v.lower() in ['outcome', 'result', 'customer reaction']:
            raise ValueError("causal_outcome must be specific, not generic")
        return v


class QueryValidationScore(BaseModel):
    """Schema for query validation results."""
    
    causal_depth: int = Field(
        ge=1,
        le=5,
        description="Root cause identification (1-5)"
    )
    clarity: int = Field(
        ge=1,
        le=5,
        description="Unambiguous phrasing (1-5)"
    )
    business_value: int = Field(
        ge=1,
        le=5,
        description="Actionable for RCA (1-5)"
    )
    aggregate_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Average of three scores"
    )
    decision: str = Field(
        pattern="^(ACCEPT|REJECT)$",
        description="Acceptance decision"
    )
    reasoning: str = Field(
        description="Brief justification for scores"
    )


class EvidenceResult(BaseModel):
    """Schema for evidence grounding results."""
    
    agent_turn: Optional[int] = Field(
        default=None,
        description="Turn number where agent action occurred"
    )
    customer_response_turn: Optional[int] = Field(
        default=None,
        description="Turn number of customer's response"
    )
    evidence_quality: str = Field(
        pattern="^(strong|moderate|weak)$",
        description="Quality rating of evidence"
    )
    grounding_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for turn matching"
    )


class EventDetectionResult(BaseModel):
    """Schema for event detection results."""
    
    event_type: str = Field(
        description="Detected event type"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for classification"
    )
    label_scores: Dict[str, float] = Field(
        description="Scores for all event labels"
    )


class TranscriptEpisode(BaseModel):
    """Schema for a processed transcript episode."""
    
    id: str = Field(
        description="Unique transcript identifier"
    )
    text: str = Field(
        description="Raw transcript text"
    )
    text_with_turns: str = Field(
        description="Turn-indexed transcript (Turn N: Speaker: Text)"
    )
    domain: str = Field(
        default="General",
        description="Business domain category"
    )
    turn_count: int = Field(
        description="Number of conversation turns"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class FinalQuery(BaseModel):
    """Schema for final output query with all metadata."""
    
    Query_ID: str
    Query: str
    Query_Category: str
    Complexity_Level: int
    Evidence_Location: str
    Remarks: str
    Validation_Score: float
    Evidence_Quality: str
    Lehnert_Category: str
