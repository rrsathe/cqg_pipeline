"""
Task 2 Recognition Module (REVISED): Extract topics from QA pairs

NOT from clusters - extract directly from Q&A content
for cleaner, more reliable topic extraction
"""

import logging
import re
from typing import List, Dict
from collections import Counter

logger = logging.getLogger(__name__)


class RecognitionModule:
    """Extract key topics from QA pairs (not clusters)"""
    
    def __init__(self):
        """Initialize recognition module"""
        self.logger = logging.getLogger(__name__)
    
    def extract_topics_from_qas(
        self,
        qa_pairs: List[Dict],
        cluster_action: str = None
    ) -> List[str]:
        """
        Extract topics directly from QA pairs.
        
        REVISED: Uses QA content, not cluster data
        
        Args:
            qa_pairs: QA pairs from other team
            cluster_action: Optional action label for context
        
        Returns:
            List of topic strings, sorted by importance
        """
        
        topics = {}
        
        # 1. Extract from queries (questions asked)
        for qa in qa_pairs:
            query = qa.get('query', '').lower()
            
            # Extract noun phrases
            words = re.findall(r'\b[a-z]{4,}\b', query)
            
            # Filter stop words
            stop_words = {
                'what', 'when', 'where', 'which', 'why', 'how',
                'does', 'have', 'been', 'really', 'would', 'could',
                'should', 'could', 'must', 'likely', 'often'
            }
            
            words = [w for w in words if w not in stop_words]
            
            # Count frequency
            for word in words[:5]:  # Top 5 words per query
                if word not in topics:
                    topics[word] = {'query_count': 0, 'response_count': 0}
                topics[word]['query_count'] += 1
        
        # 2. Extract from responses (answers provided)
        for qa in qa_pairs:
            response = qa.get('response', '').lower()
            
            # Extract noun phrases and key terms
            words = re.findall(r'\b[a-z]{4,}\b', response)
            
            # Filter more aggressively for responses
            stop_words = {
                'from', 'have', 'been', 'when', 'where', 'which',
                'reason', 'caused', 'because', 'result', 'outcome',
                'percent', 'percentage', 'time', 'days', 'hours',
                'includes', 'involves', 'related', 'based'
            }
            
            words = [w for w in words if w not in stop_words]
            
            # Count frequency
            for word in words[:5]:  # Top 5 words per response
                if word not in topics:
                    topics[word] = {'query_count': 0, 'response_count': 0}
                topics[word]['response_count'] += 1
        
        # 3. Add cluster action if provided
        if cluster_action:
            action = cluster_action.lower().strip()
            if action not in topics:
                topics[action] = {'query_count': 2, 'response_count': 2}
            else:
                topics[action]['query_count'] += 2
                topics[action]['response_count'] += 2
        
        # 4. Score topics by combined frequency
        scored_topics = []
        for topic, counts in topics.items():
            total_count = counts['query_count'] + counts['response_count']
            score = (counts['query_count'] * 0.6) + (counts['response_count'] * 0.4)
            
            scored_topics.append({
                'topic': topic,
                'score': score,
                'total_count': total_count
            })
        
        # Sort by score and return top topics
        sorted_topics = sorted(scored_topics, key=lambda x: x['score'], reverse=True)
        result = [t['topic'] for t in sorted_topics[:8]]
        
        self.logger.info(f"Extracted {len(result)} topics from QA pairs: {result}")
        
        return result
    
    def extract_key_phrases_from_qas(
        self,
        qa_pairs: List[Dict]
    ) -> List[str]:
        """
        Extract multi-word key phrases from QA pairs.
        
        More sophisticated approach using n-grams.
        
        Args:
            qa_pairs: QA pairs from other team
        
        Returns:
            List of key phrases (2-3 words)
        """
        
        phrases = Counter()
        
        for qa in qa_pairs:
            query = qa.get('query', '').lower()
            response = qa.get('response', '')[:200].lower()  # First 200 chars
            
            combined = query + " " + response
            
            # Find 2-3 word phrases
            words = re.findall(r'\b[a-z]{3,}\b', combined)
            
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                phrases[phrase] += 1
            
            if len(words) > 2:
                for i in range(len(words) - 2):
                    phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                    phrases[phrase] += 1
        
        # Return top phrases
        top_phrases = [phrase for phrase, _ in phrases.most_common(5)]
        
        return top_phrases
