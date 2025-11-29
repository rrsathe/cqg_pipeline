"""
Main pipeline runner coordinating all CQG components.

Orchestrates the full workflow:
1. Load transcripts
2. Action-Driven Semantic Clustering (Cluster & Characterize)
3. Generate diverse causal queries
4. Export results
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from .transcript_loader import TranscriptLoader
from .action_extractor import ActionDrivenExtractor
from .cluster_query_generator import ClusterQueryGenerator
from .llm_client import LLMClient
from .config import Config

logger = logging.getLogger(__name__)


class CQGPipeline:
    """
    Causal Query Generation Pipeline.
    
    Coordinates all components to generate high-quality causal queries
    from contact center transcripts using Action-Driven Semantic Clustering.
    """
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize pipeline with all components.
        
        Args:
            llm_model: Model for query generation (default from config)
        """
        logger.info("=" * 80)
        logger.info("Initializing Causal Query Generation Pipeline (Action-Driven)")
        logger.info("=" * 80)
        
        # Initialize components
        self.loader = TranscriptLoader()
        
        # Create LLM client
        llm_model = llm_model or Config.LLM_MODEL
        self.llm_client = LLMClient(model=llm_model)
        
        # New components
        self.extractor = ActionDrivenExtractor()
        self.generator = ClusterQueryGenerator(llm_client=self.llm_client)
        
        # Results storage
        self.queries: List[str] = []
        self.cluster_chars: Dict = {}
        
        # Statistics
        self.stats = {
            "ingested": 0,
            "clusters_found": 0,
            "queries_generated": 0,
            "start_time": None,
            "end_time": None,
            "duration_seconds": 0
        }
        
        logger.info(f"✅ LLM Model: {llm_model}")
        logger.info("✅ All components initialized")
    
    def run(
        self,
        transcript_file: str,
        output_csv: Optional[str] = None,
        output_metrics: Optional[str] = None,
        max_transcripts: Optional[int] = None
    ) -> List[str]:
        """
        Execute full CQG pipeline.
        
        Args:
            transcript_file: Path to input JSON/CSV
            output_csv: Path for output CSV (default from config)
            output_metrics: Path for metrics JSON (default from config)
            max_transcripts: Limit number of transcripts to process (not used in clustering load yet, but kept for API compat)
        
        Returns:
            List of generated queries
        """
        self.stats["start_time"] = datetime.now()
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING CAUSAL QUERY GENERATION PIPELINE")
        logger.info("=" * 80)
        
        # Set defaults
        output_csv = output_csv or str(Config.DEFAULT_OUTPUT_CSV)
        output_metrics = output_metrics or str(Config.DEFAULT_METRICS_JSON)
        
        # Step 1: Action-Driven Semantic Clustering
        # Note: extractor.analyze_transcripts handles loading internally for now
        logger.info(f"🔄 Phase 1: Analyzing transcripts from {transcript_file}...")
        self.cluster_chars = self.extractor.analyze_transcripts(transcript_file)
        
        self.stats["clusters_found"] = len(self.cluster_chars)
        logger.info(f"✅ Found {self.stats['clusters_found']} action clusters")
        
        if not self.cluster_chars:
            logger.error("No clusters found. Exiting.")
            return []
            
        # Step 2: Query Generation
        logger.info(f"🔄 Phase 2: Generating queries...")
        self.queries = self.generator.generate_queries(self.cluster_chars)
        
        self.stats["queries_generated"] = len(self.queries)
        logger.info(f"✅ Generated {self.stats['queries_generated']} queries")
        
        # Step 3: Export results
        self._export_results(output_csv, output_metrics)
        
        # Finalize
        self.stats["end_time"] = datetime.now()
        self.stats["duration_seconds"] = (
            self.stats["end_time"] - self.stats["start_time"]
        ).total_seconds()
        
        self._print_summary()
        
        return self.queries
    
    def _export_results(self, output_csv: str, output_metrics: str) -> None:
        """Export queries to CSV and metrics to JSON."""
        logger.info("\n" + "=" * 80)
        logger.info("EXPORTING RESULTS")
        logger.info("=" * 80)
        
        # Ensure output directory exists
        Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        if self.queries:
            # Export CSV
            df = pd.DataFrame(self.queries, columns=["Query"])
            df.to_csv(output_csv, index=False)
            logger.info(f"✅ Exported {len(self.queries)} queries to: {output_csv}")
            
            # Export metrics
            metrics = self.stats.copy()
            # Add cluster summaries to metrics
            metrics["clusters"] = {
                cid: {
                    "label": data["action_label"],
                    "size": data["size"],
                    "summary": data["summary"]
                }
                for cid, data in self.cluster_chars.items()
            }
            
            with open(output_metrics, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"✅ Exported metrics to: {output_metrics}")
        else:
            logger.warning("⚠️  No queries generated. No CSV exported.")
    
    def _print_summary(self) -> None:
        """Print execution summary."""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 Clusters Found: {self.stats['clusters_found']}")
        logger.info(f"📈 Queries Generated: {self.stats['queries_generated']}")
        logger.info(f"⏱️  Execution Time: {self.stats['duration_seconds']:.1f}s")
        logger.info("\n" + "=" * 80)
