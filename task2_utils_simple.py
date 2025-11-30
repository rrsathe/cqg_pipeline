"""
Task 2 Utility Functions: Load QAs, link to clusters, export results
"""

import json
import csv
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_qa_pairs(qa_file_path: str) -> List[Dict]:
    """Load QA pairs from JSON file"""
    
    try:
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_pairs = json.load(f)
        
        if not isinstance(qa_pairs, list):
            logger.warning("QA pairs should be a JSON list")
            return []
        
        logger.info(f"✅ Loaded {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    except FileNotFoundError:
        logger.error(f"❌ QA file not found: {qa_file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing JSON: {e}")
        return []


def link_qas_to_clusters(
    cluster_chars: Dict,
    qa_pairs: List[Dict]
) -> Dict[int, List[Dict]]:
    """Link QA pairs to clusters by action_label matching"""
    
    cluster_qa_map = {}
    
    for cid, cluster_data in cluster_chars.items():
        action_label = cluster_data.get('action_label', '').lower().strip()
        key_phrases = [kp.lower().strip() for kp in cluster_data.get('key_phrases', [])]
        
        linked_qas = []
        
        for qa in qa_pairs:
            query = qa.get('query', '').lower()
            response = qa.get('response', '').lower()
            
            # Match by action_label in query
            if action_label and action_label in query:
                linked_qas.append(qa)
            
            # Match by key phrases in query
            elif any(kp in query for kp in key_phrases if kp):
                linked_qas.append(qa)
            
            # Match by action_label in response
            elif action_label and action_label in response:
                linked_qas.append(qa)
        
        # Remove duplicates
        linked_qas = list({qa['query']: qa for qa in linked_qas}.values())
        
        cluster_qa_map[cid] = linked_qas
    
    return cluster_qa_map


def save_followups_csv(
    enriched_clusters: Dict,
    output_path: str
) -> None:
    """Save follow-up questions to CSV"""
    
    rows = []
    
    for cid, cluster_data in enriched_clusters.items():
        action_label = cluster_data.get('action_label', 'unknown')
        followups = cluster_data.get('deep_followups', [])
        topics = ', '.join(cluster_data.get('topics', [])[:3])
        
        for i, followup_dict in enumerate(followups, 1):
            if isinstance(followup_dict, dict):
                question = followup_dict.get('question', '')
                category = followup_dict.get('category', 'other')
            else:
                question = followup_dict
                category = 'other'
            
            rows.append({
                'cluster_id': cid,
                'action_label': action_label,
                'question_number': i,
                'question': question,
                'category': category,
                'related_topics': topics
            })
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            fieldnames = [
                'cluster_id',
                'action_label',
                'question_number',
                'question',
                'category',
                'related_topics'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            logger.info(f"✅ Saved {len(rows)} follow-up questions to {output_path}")
