import json
import os
import argparse
import numpy as np
import faiss
import torch
from typing import Dict, List, Optional, Any
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
from tqdm import tqdm
from .config import Config

class ActionDrivenExtractor:
    def __init__(self, model_name=Config.EMBEDDING_MODEL):
        """
        Initialize the extractor with a SentenceTransformer model.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        print(f"Initializing ActionDrivenExtractor with model '{model_name}' on {device}...")
        self.model = SentenceTransformer(model_name, device=device)

        # Pre-compute ontology embeddings
        self.ACTION_ONTOLOGY = {
            'inform': 'provide information explain clarify',
            'request': 'ask inquire need require assistance',
            'apologize': 'sorry apologize regret inconvenience',
            'resolve': 'fixed solved working now',
            'redirect': 'transfer escalate manager'
        }
        print("Caching ontology embeddings...")
        self.ontology_keys = list(self.ACTION_ONTOLOGY.keys())
        self.ontology_values = list(self.ACTION_ONTOLOGY.values())
        self.ontology_embeddings = self.model.encode(self.ontology_values)

    def analyze_transcripts(
        self, 
        json_path: str,
        outcomes: Optional[Dict] = None,
        domain: Optional[str] = None
    ) -> Dict:
        """
        Complete clustering + characterization pipeline
        
        Returns:
        {
            cluster_id: {
                "action_label": str,
                "exemplars": List[str],
                "key_phrases": List[str],
                "tone": str,
                "speaker_distribution": Dict,
                "size": int,
                "success_rate": float,
                "centroid": np.ndarray,
                "summary": str
            }
        }
        """
        # Load transcripts
        utterances = self.load_transcripts(json_path, domain=domain)
        if not utterances:
            return {}
            
        texts = [u['text'] for u in utterances]
        
        # Embed
        embeddings = self.embed_utterances(texts)
        
        # Cluster
        # Ensure we don't ask for more clusters than samples
        n_clusters = min(Config.N_CLUSTERS, len(texts))
        if n_clusters < 2:
            n_clusters = 1
            
        labels, centers = self.cluster_actions(
            embeddings, 
            n_clusters=n_clusters
        )
        
        # Characterize
        cluster_chars = self._characterize_all_clusters(
            embeddings, 
            labels, 
            texts, 
            utterances,
            outcomes
        )
        
        return cluster_chars

    def load_transcripts(self, json_path, domain: Optional[str] = None):
        """
        Load transcripts from a JSON file.
        Adapted from extract_trajectories.py to handle various formats including
        consolidated transcripts.
        """
        print(f"Loading transcripts from {json_path}...")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(json_path, 'r') as reader:
            data = json.load(reader)

        utterances = []
        DEFAULT_USER_ALIASES = ["user", "customer", "client"]
        DEFAULT_USER_NAME = "user"
        DEFAULT_SYS_NAME = "system"

        # Helper to process a single turn
        def process_turn(t):
            text = None
            # try common text fields
            for key in ("text", "utterance", "transcript", "Content", "content", "utterances", "utterance_text"):
                if isinstance(t, dict) and key in t:
                    text = t[key]
                    break
            
            # try common speaker/role fields
            speaker_raw = None
            for key in ("speaker", "ParticipantRole", "participant_role", "role", "speaker_label"):
                if isinstance(t, dict) and key in t:
                    speaker_raw = t[key]
                    break

            if text is None and isinstance(t, str):
                text = t

            if text:
                speaker = DEFAULT_USER_NAME if (isinstance(speaker_raw, str) and speaker_raw.lower() in DEFAULT_USER_ALIASES) else DEFAULT_SYS_NAME
                utterances.append({
                    "text": text,
                    "speaker": speaker
                })

        # Logic adapted from extract_trajectories.py get_json_dialog
        turns = []
        
        # Case 1: List of dialogues (consolidated file) or List of turns
        if isinstance(data, list):
            if not data:
                return []
            
            first_item = data[0]
            if isinstance(first_item, dict):
                # Check if it's a list of turns directly (Amazon Transcribe style but just the list)
                # or if it's a list of dialogues (has 'conversation', 'transcript_id', etc.)
                
                is_dialogue_list = False
                # Check for keys that strongly suggest a dialogue object
                if "conversation" in first_item or "transcript_id" in first_item or "turns" in first_item or "transcript" in first_item:
                    is_dialogue_list = True
                
                if is_dialogue_list:
                     # It's a list of dialogues, we'll aggregate ALL turns from ALL dialogues
                     print(f"Detected list of {len(data)} dialogues.")
                     filtered_count = 0
                     for entry in data:
                         if isinstance(entry, dict):
                             # Domain filtering
                             if domain:
                                 entry_domain = entry.get("domain") or entry.get("Domain")
                                 if not entry_domain or entry_domain.lower() != domain.lower():
                                     continue
                             
                             filtered_count += 1
                             # Try to find the conversation list
                             conv = entry.get("conversation") or entry.get("turns") or entry.get("transcript")
                             if conv:
                                 for t in conv:
                                     process_turn(t)
                     
                     if domain:
                         print(f"Filtered to {filtered_count} dialogues matching domain '{domain}'.")
                     return utterances
                elif "Content" in first_item or "text" in first_item:
                     # Assume it's a single dialogue's list of turns
                     turns = data
                else:
                    # Fallback: try to see if it looks like a dialogue object anyway
                    pass
            else:
                 # It might be a list of strings?
                 pass

        # Case 2: Dict with "Transcript" (Amazon Transcribe)
        elif isinstance(data, dict):
            if "Transcript" in data:
                turns = data["Transcript"]
            elif "dialogs" in data:
                 # Dict of dialog_id -> turns
                 for d_id, d_turns in data["dialogs"].items():
                     for t in d_turns:
                         process_turn(t)
                 return utterances
            else:
                # Try to find any list value that looks like turns
                for v in data.values():
                    if isinstance(v, list) and v and (isinstance(v[0], dict) or isinstance(v[0], str)):
                        turns = v
                        break

        if turns:
            for t in turns:
                process_turn(t)
        
        return utterances

    def embed_utterances(self, texts):
        """
        Embed a list of utterance texts.
        """
        print(f"Embedding {len(texts)} utterances...")
        embeddings = self.model.encode(texts, batch_size=128, show_progress_bar=True)
        
        # GloVe zero-vector fix
        zero_vectors = np.where(~np.any(embeddings, axis=1))[0]
        if len(zero_vectors) > 0:
            print(f"Fixing {len(zero_vectors)} zero vectors...")
            embeddings[zero_vectors, 0] = 1
            
        return embeddings

    def cluster_actions(self, embeddings, n_clusters):
        """
        Cluster embeddings using KMeans.
        """
        print(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        return kmeans.labels_, kmeans.cluster_centers_

    def _characterize_all_clusters(
        self, 
        embeddings, 
        labels, 
        texts, 
        utterances,
        outcomes
    ) -> Dict:
        """Enrich each cluster with metadata"""
        cluster_chars = {}
        
        for cluster_id in range(labels.max() + 1):
            mask = labels == cluster_id
            cluster_texts = [t for t, m in zip(texts, mask) if m]
            cluster_embeddings = embeddings[mask]
            cluster_utterances = [u for u, m in zip(utterances, mask) if m]
            
            if not cluster_texts:
                continue

            # Get exemplars
            exemplars = self._get_top_exemplars(
                cluster_embeddings, 
                cluster_texts, 
                k=Config.N_EXEMPLARS
            )
            
            # Extract features
            key_phrases = self._extract_key_phrases(cluster_texts)
            tone = self._classify_tone(cluster_texts)
            speaker_dist = self._analyze_speakers(cluster_utterances)
            success_rate = self._compute_success_rate(
                cluster_utterances, 
                outcomes
            ) if outcomes else None
            
            action_label = self.infer_action_label(exemplars[0])
            
            cluster_chars[cluster_id] = {
                "action_label": action_label,
                "exemplars": exemplars,
                "key_phrases": key_phrases,
                "tone": tone,
                "speaker_distribution": speaker_dist,
                "size": len(cluster_texts),
                "success_rate": success_rate,
                "centroid": cluster_embeddings.mean(axis=0),
                "summary": self._summarize_cluster(action_label, exemplars, key_phrases)
            }
        
        return cluster_chars

    def _get_top_exemplars(self, cluster_embeddings, cluster_texts, k=5):
        """Find the closest utterances to the cluster centroid."""
        centroid = cluster_embeddings.mean(axis=0).reshape(1, -1)
        sims = cosine_similarity(cluster_embeddings, centroid).flatten()
        # Get indices of top k most similar
        top_k_indices = sims.argsort()[-k:][::-1]
        return [cluster_texts[i] for i in top_k_indices]

    def _extract_key_phrases(self, texts, top_n=5):
        """Extract top TF-IDF terms."""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            # Sum tfidf scores for each term
            sums = tfidf_matrix.sum(axis=0)
            # Sort by score
            data = []
            for col, term in enumerate(feature_names):
                data.append((term, sums[0, col]))
            ranking = sorted(data, key=lambda x: x[1], reverse=True)
            return [term for term, score in ranking[:top_n]]
        except ValueError:
            # Can happen if empty vocabulary or stop words remove everything
            return []

    def _classify_tone(self, texts):
        """Classify tone using TextBlob on a sample."""
        sample_size = min(len(texts), Config.SENTIMENT_SAMPLE_SIZE)
        sample = texts[:sample_size] # Simple sampling, could be random
        scores = [TextBlob(t).sentiment.polarity for t in sample]
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.1:
            return "positive"
        elif avg_score < -0.1:
            return "negative"
        else:
            return "neutral"

    def _analyze_speakers(self, utterances):
        """Analyze speaker distribution."""
        counts = {}
        for u in utterances:
            s = u['speaker'].lower()
            counts[s] = counts.get(s, 0) + 1
        return counts

    def _compute_success_rate(self, utterances, outcomes):
        """Compute success rate if outcomes are available."""
        # Placeholder: requires linking utterances to transcript IDs and then to outcomes
        # Since utterances here don't have transcript IDs attached in the current load_transcripts,
        # we would need to enhance load_transcripts to include metadata.
        # For now, returning None or 0.0
        return None

    def _summarize_cluster(self, label, exemplars, key_phrases):
        """Create a human-readable summary."""
        return f"Action: {label}. Key themes: {', '.join(key_phrases)}. Example: '{exemplars[0]}'"

    def infer_action_label(self, exemplar_text):
        """
        Infer action label using cached ontology embeddings.
        """
        ex_emb = self.model.encode([exemplar_text])
        sims = cosine_similarity(ex_emb, self.ontology_embeddings)[0]
        best_idx = np.argmax(sims)
        return self.ontology_keys[best_idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action-Driven Semantic Clustering")
    parser.add_argument("--input", type=str, default="data/final_transcripts_domain_corrected.json", help="Path to input JSON file")
    parser.add_argument("--clusters", type=int, default=Config.N_CLUSTERS, help="Number of clusters")
    parser.add_argument("--domain", type=str, default=None, help="Filter transcripts by domain")
    
    args = parser.parse_args()

    try:
        extractor = ActionDrivenExtractor()
        results = extractor.analyze_transcripts(args.input, domain=args.domain)
        
        print("\nDiscovered Clusters:")
        for cid, data in results.items():
            print(f"Cluster {cid} ({data['action_label']}): {data['size']} utterances")
            print(f"  Summary: {data['summary']}")
            print(f"  Tone: {data['tone']}")
            print(f"  Speakers: {data['speaker_distribution']}")
            print("-" * 40)
            
    except Exception as e:
        print(f"An error occurred: {e}")
