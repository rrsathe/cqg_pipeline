"""
Evidence grounding using fuzzy string matching.

Maps generated queries back to specific turn indices in transcripts
using RapidFuzz for fast approximate matching.
"""

import logging
import string
from typing import List, Dict, Optional

from rapidfuzz import fuzz

from .schemas import EvidenceResult
from .config import Config

logger = logging.getLogger(__name__)


class EvidenceGrounder:
    """
    Ground queries to transcript turns using fuzzy matching.
    
    Algorithm:
    1. Parse turns from "Turn N: Speaker: Text" format
    2. Extract keywords from query
    3. Find best matching agent turn (RapidFuzz)
    4. Find next customer response turn
    5. Calculate confidence and quality rating
    """
    
    def __init__(self, threshold: Optional[int] = None):
        """
        Initialize evidence grounder.
        
        Args:
            threshold: Minimum fuzzy match score (0-100), default from config
        """
        self.threshold = threshold or Config.FUZZY_MATCH_THRESHOLD
        self.stop_words = Config.STOP_WORDS
    
    def extract_evidence(
        self,
        query: str,
        transcript_with_turns: str
    ) -> EvidenceResult:
        """
        Extract evidence location for query from transcript.
        
        Args:
            query: Generated causal query
            transcript_with_turns: Turn-indexed transcript
        
        Returns:
            EvidenceResult with turn numbers and confidence
        """
        # Parse turns
        turns = self._parse_turns(transcript_with_turns)
        
        if not turns:
            logger.warning("No turns found in transcript")
            return EvidenceResult(
                agent_turn=None,
                customer_response_turn=None,
                evidence_quality="weak",
                grounding_confidence=0.0
            )
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        if not keywords:
            logger.warning("No keywords extracted from query")
            return EvidenceResult(
                agent_turn=None,
                customer_response_turn=None,
                evidence_quality="weak",
                grounding_confidence=0.0
            )
        
        # Find best matching agent turn
        agent_turn = self._find_best_turn(keywords, turns, speaker="Agent")
        
        if not agent_turn:
            logger.debug("No agent turn found above threshold")
            return EvidenceResult(
                agent_turn=None,
                customer_response_turn=None,
                evidence_quality="weak",
                grounding_confidence=0.0
            )
        
        # Find customer response after agent turn
        customer_turn = self._find_next_customer_turn(agent_turn['number'], turns)
        
        # Calculate confidence
        agent_score = agent_turn.get('match_score', 0)
        customer_score = customer_turn.get('match_score', 0) if customer_turn else 0
        confidence = (agent_score + customer_score) / 200.0  # Normalize to 0-1
        
        # Rate quality
        quality = self._rate_quality(agent_score)
        
        return EvidenceResult(
            agent_turn=agent_turn['number'],
            customer_response_turn=customer_turn['number'] if customer_turn else None,
            evidence_quality=quality,
            grounding_confidence=min(confidence, 1.0)
        )
    
    def _parse_turns(self, transcript: str) -> List[Dict]:
        """
        Parse "Turn N: Speaker: Text" format into structured turns.
        
        Args:
            transcript: Turn-indexed transcript text
        
        Returns:
            List of turn dicts with {number, speaker, text}
        """
        turns = []
        
        for line in transcript.split('\n'):
            line = line.strip()
            if not line or not line.startswith('Turn '):
                continue
            
            try:
                # Parse "Turn N: Speaker: Text"
                parts = line.split(':', 3)
                if len(parts) >= 3:
                    # Extract turn number
                    turn_num_str = parts[0].replace('Turn ', '').strip()
                    turn_num = int(turn_num_str)
                    
                    # Extract speaker
                    speaker = parts[1].strip()
                    
                    # Extract text
                    text = parts[2].strip() if len(parts) == 3 else parts[3].strip()
                    
                    turns.append({
                        'number': turn_num,
                        'speaker': speaker,
                        'text': text
                    })
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse turn: {line[:50]}... ({e})")
                continue
        
        return turns
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query.
        
        Args:
            query: Query text
        
        Returns:
            List of keywords (lowercased)
        """
        # Simple tokenization
        words = query.lower().split()
        
        # Filter stop words and short words
        keywords = [
            w for w in words
            if w not in self.stop_words and len(w) > 3
        ]
        
        # Remove punctuation safely
        keywords = [w.strip(string.punctuation) for w in keywords]
        
        return [w for w in keywords if w]  # Remove empty strings
    
    def _find_best_turn(
        self,
        keywords: List[str],
        turns: List[Dict],
        speaker: str
    ) -> Optional[Dict]:
        """
        Find turn with best keyword match for given speaker.
        
        Args:
            keywords: List of keywords to match
            turns: List of turn dicts
            speaker: Speaker to filter by (e.g., "Agent", "Customer")
        
        Returns:
            Best matching turn dict with 'match_score' added, or None
        """
        best_turn = None
        best_score = 0
        
        for turn in turns:
            # Filter by speaker
            if speaker.lower() not in turn['speaker'].lower():
                continue
            
            # Fuzzy match keywords against turn text
            turn_text = turn['text'].lower()
            
            # Find best matching keyword
            max_match = max(
                [fuzz.partial_ratio(kw, turn_text) for kw in keywords],
                default=0
            )
            
            if max_match > best_score:
                best_score = max_match
                best_turn = turn.copy()
        
        # Check threshold
        if best_turn and best_score >= self.threshold:
            best_turn['match_score'] = best_score
            return best_turn
        
        return None
    
    def _find_next_customer_turn(
        self,
        agent_turn_num: int,
        turns: List[Dict]
    ) -> Optional[Dict]:
        """
        Find first customer turn after agent turn.
        
        Args:
            agent_turn_num: Agent turn number
            turns: List of turn dicts
        
        Returns:
            Customer turn dict with 'match_score' added, or None
        """
        for turn in turns:
            if (turn['number'] > agent_turn_num and
                'customer' in turn['speaker'].lower()):
                # Exact next turn - high confidence
                turn = turn.copy()
                turn['match_score'] = 100
                return turn
        
        return None
    
    def _rate_quality(self, match_score: int) -> str:
        """
        Rate evidence quality based on match score.
        
        Args:
            match_score: Fuzzy match score (0-100)
        
        Returns:
            Quality rating: "strong", "moderate", or "weak"
        """
        if match_score >= 95:
            return "strong"
        elif match_score >= 80:
            return "moderate"
        else:
            return "weak"
    
    def extract_evidence_batch(
        self,
        queries: List[str],
        transcripts: List[str]
    ) -> List[EvidenceResult]:
        """
        Extract evidence for multiple queries.
        
        Args:
            queries: List of query strings
            transcripts: Corresponding turn-indexed transcripts
        
        Returns:
            List of EvidenceResult objects
        """
        results = []
        
        for query, transcript in zip(queries, transcripts):
            result = self.extract_evidence(query, transcript)
            results.append(result)
        
        return results
    
    def format_evidence_location(self, evidence: EvidenceResult) -> str:
        """
        Format evidence location as string for CSV output.
        
        Args:
            evidence: EvidenceResult object
        
        Returns:
            Formatted string like "Turns 8-9" or "N/A"
        """
        if evidence.agent_turn is None:
            return "N/A"
        
        if evidence.customer_response_turn is not None:
            return f"Turns {evidence.agent_turn}-{evidence.customer_response_turn}"
        else:
            return f"Turn {evidence.agent_turn}"
