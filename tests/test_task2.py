import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from cqg_pipeline.pipeline_runner import CQGPipeline
from cqg_pipeline.config import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


def test_task2():
    print("Testing Task 2 Pipeline...")
    
    # Initialize pipeline
    pipeline = CQGPipeline()
    
    # Mock Task 1 output (clusters)
    pipeline.cluster_chars = {
        0: {
            'action_label': 'refund_request',
            'key_phrases': ['refund', 'billing', 'charge'],
            'size': 10,
            'summary': 'User asking for refund due to billing error'
        },
        1: {
            'action_label': 'subscription_cancellation',
            'key_phrases': ['cancel', 'subscription', 'stop'],
            'size': 5,
            'summary': 'User wants to cancel subscription'
        }
    }
    
    # Run Task 2
    qa_file = Path(__file__).parent / "dummy_qa_pairs.json"
    output_csv = Path(__file__).parent / "test_task2_output.csv"
    
    pipeline.run_task2_followupqg_llm_knowledge(
        qa_input_path=str(qa_file),
        output_followups_csv=str(output_csv)
    )
    
    # Verify output
    if output_csv.exists():
        print(f"✅ Output CSV created at {output_csv}")
        with open(output_csv, 'r') as f:
            print("Output content:")
            print(f.read())
    else:
        print("❌ Output CSV not created")

if __name__ == "__main__":
    test_task2()
