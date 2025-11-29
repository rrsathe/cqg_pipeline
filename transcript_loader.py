"""
Transcript loader and preprocessor.

Parses diverse transcript formats (CSV, JSON arrays) and normalizes them
into turn-indexed episodes for downstream processing.
"""

import ast
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from .schemas import TranscriptEpisode
from .config import Config

logger = logging.getLogger(__name__)


class TranscriptLoader:
    """
    Load and normalize transcripts from CSV files.
    
    Handles multiple conversation formats:
    - JSON array string: '[{"speaker": "Agent", "text": "..."}]'
    - Plain text with speaker labels: "Agent: ... Customer: ..."
    - Pre-formatted turn-indexed text
    
    Outputs turn-indexed episodes for downstream components.
    """
    
    def __init__(self):
        self.episodes: List[TranscriptEpisode] = []
        self.stats = {
            "total_loaded": 0,
            "parse_errors": 0,
            "malformed_rows": []
        }
    
    def load(self, file_path: str) -> List[TranscriptEpisode]:
        """
        Load transcripts from CSV file.
        
        Args:
            file_path: Path to CSV file with columns:
                - transcript_id (or similar identifier)
                - conversation (JSON array or text)
                - domain (optional)
        
        Returns:
            List of TranscriptEpisode objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
        
        logger.info(f"Loading transcripts from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise
        
        logger.info(f"CSV loaded: {len(df)} rows")
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                episode = self._parse_row(row, idx)
                if episode:
                    self.episodes.append(episode)
                    self.stats["total_loaded"] += 1
            except Exception as e:
                logger.warning(f"Failed to parse row {idx}: {e}")
                self.stats["parse_errors"] += 1
                self.stats["malformed_rows"].append({
                    "row_index": idx,
                    "error": str(e)
                })
        
        logger.info(
            f"✅ Loaded {self.stats['total_loaded']} transcripts "
            f"({self.stats['parse_errors']} errors)"
        )
        
        return self.episodes
    
    def _parse_row(self, row: pd.Series, idx: int) -> Optional[TranscriptEpisode]:
        """
        Parse a single CSV row into a TranscriptEpisode.
        
        Args:
            row: Pandas Series representing one CSV row
            idx: Row index for default ID
        
        Returns:
            TranscriptEpisode or None if parsing fails
        """
        # Extract transcript ID
        transcript_id = self._extract_id(row, idx)
        
        # Extract conversation
        conversation_raw = self._extract_conversation(row)
        if not conversation_raw:
            logger.warning(f"Row {idx}: Empty conversation")
            return None
        
        # Parse conversation format
        conversation_data = self._parse_conversation_format(conversation_raw)
        
        # Build turn-indexed text
        text_with_turns, raw_text, turn_count = self._build_turn_indexed_text(
            conversation_data
        )
        
        # Extract domain
        domain = row.get('domain', row.get('Domain', 'General'))
        if pd.isna(domain):
            domain = 'General'
        
        # Build episode
        episode = TranscriptEpisode(
            id=transcript_id,
            text=raw_text,
            text_with_turns=text_with_turns,
            domain=str(domain),
            turn_count=turn_count,
            metadata={
                "row_index": idx,
                "raw_format": type(conversation_data).__name__
            }
        )
        
        return episode
    
    def _extract_id(self, row: pd.Series, idx: int) -> str:
        """Extract transcript ID from row."""
        for col in ['transcript_id', 'Transcript_ID', 'id', 'ID']:
            if col in row and not pd.isna(row[col]):
                return str(row[col])
        
        # Default ID
        return f"T_{idx:04d}"
    
    def _extract_conversation(self, row: pd.Series) -> Optional[str]:
        """Extract conversation field from row."""
        for col in ['conversation', 'Conversation', 'transcript', 'Transcript', 'text', 'Text']:
            if col in row and not pd.isna(row[col]):
                return str(row[col])
        
        return None
    
    def _parse_conversation_format(self, conversation_raw: str) -> Any:
        """
        Parse conversation string into structured format.
        
        Handles:
        - JSON array string
        - Plain text
        
        Returns:
            List of dicts or string
        """
        # Try parsing as JSON array
        if conversation_raw.strip().startswith('['):
            try:
                # Try ast.literal_eval first (safer)
                data = ast.literal_eval(conversation_raw)
                if isinstance(data, list):
                    return data
            except (ValueError, SyntaxError):
                pass
            
            try:
                # Try json.loads
                data = json.loads(conversation_raw)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        
        # Return as plain text
        return conversation_raw
    
    def _build_turn_indexed_text(
        self,
        conversation_data: Any
    ) -> tuple[str, str, int]:
        """
        Build turn-indexed text from conversation data.
        
        Args:
            conversation_data: List of dicts or string
        
        Returns:
            Tuple of (turn_indexed_text, raw_text, turn_count)
        """
        if isinstance(conversation_data, list):
            # List of turn dicts
            turns = []
            raw_parts = []
            
            for i, turn in enumerate(conversation_data, 1):
                if isinstance(turn, dict):
                    speaker = turn.get('speaker', turn.get('Speaker', 'Unknown'))
                    text = turn.get('text', turn.get('Text', ''))
                else:
                    speaker = 'Unknown'
                    text = str(turn)
                
                turns.append(f"Turn {i}: {speaker}: {text}")
                raw_parts.append(f"{speaker}: {text}")
            
            turn_indexed = "\n".join(turns)
            raw_text = "\n".join(raw_parts)
            turn_count = len(turns)
            
        else:
            # Plain text - try to identify turns
            text = str(conversation_data)
            turns = self._split_plain_text_into_turns(text)
            
            if turns:
                turn_indexed = "\n".join(
                    f"Turn {i}: {turn}" for i, turn in enumerate(turns, 1)
                )
                raw_text = "\n".join(turns)
                turn_count = len(turns)
            else:
                # Single turn
                turn_indexed = f"Turn 1: Unknown: {text}"
                raw_text = text
                turn_count = 1
        
        return turn_indexed, raw_text, turn_count
    
    def _split_plain_text_into_turns(self, text: str) -> List[str]:
        """
        Split plain text into turns based on speaker labels.
        
        Args:
            text: Plain text with speaker labels
        
        Returns:
            List of turn strings (Speaker: text)
        """
        # Common speaker patterns
        speaker_patterns = [
            "Agent:",
            "Customer:",
            "Representative:",
            "User:",
            "Caller:",
            "Support:",
        ]
        
        turns = []
        current_turn = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if line starts with speaker label
            is_new_turn = any(line.startswith(pattern) for pattern in speaker_patterns)
            
            if is_new_turn:
                # Save previous turn
                if current_turn:
                    turns.append(" ".join(current_turn))
                # Start new turn
                current_turn = [line]
            else:
                # Continue current turn
                current_turn.append(line)
        
        # Save last turn
        if current_turn:
            turns.append(" ".join(current_turn))
        
        return turns
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics."""
        return self.stats.copy()
