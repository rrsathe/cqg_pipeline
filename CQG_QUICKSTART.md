# Causal Query Generation (CQG) Pipeline Quickstart

This pipeline generates high-quality, diverse causal queries from contact center transcripts using **Action-Driven Semantic Clustering**.

## How It Works
1.  **Action Clustering**: Groups customer/agent turns into semantic clusters (e.g., "Billing Disputes", "Technical Issues").
2.  **Characterization**: Analyzes each cluster for tone, key phrases, and speaker dynamics.
3.  **Query Generation**: Uses LLM to generate specific causal queries based on cluster insights.

## Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Requires `python-dotenv`, `langchain-groq`, `sentence-transformers`, `scikit-learn`, `pandas`, `textblob`)*

2.  **Configure Environment**
    Ensure your `.env` file has a valid API key (Groq recommended for speed):
    ```bash
    GROQ_API_KEY=your_key_here
    ```

## Running the Pipeline

Run the pipeline using the runner script:

```bash
python -m src.cqg_pipeline.pipeline_runner
```

### Custom Usage
You can also run it programmatically:

```python
from src.cqg_pipeline import CQGPipeline

pipeline = CQGPipeline()
queries = pipeline.run(
    transcript_file="data/transcripts_with_domains.csv",
    output_csv="output/generated_queries.csv"
)
print(f"Generated {len(queries)} queries!")
```

## Output
- **Queries**: `output/task1_queries.csv` (List of generated queries)
- **Metrics**: `output/task1_metrics.json` (Cluster stats, execution time)

## Configuration
Adjust settings in `src/cqg_pipeline/config.py`:
- `N_CLUSTERS`: Number of action clusters to find (default: 8).
- `N_QUERIES_PER_SEED`: Queries generated per cluster theme.
- `LLM_MODEL`: Model to use (default: `groq/llama-3.3-70b-versatile`).
