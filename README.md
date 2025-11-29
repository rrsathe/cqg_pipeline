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

### Filter by Domain
You can filter transcripts by a specific domain using the `--domain` argument:

```bash
python -m src.cqg_pipeline.pipeline_runner --domain Hotel
```

This will filter the transcripts to only include dialogues matching the specified domain (e.g., "Filtered to 3341 dialogues matching domain 'Hotel'").

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

## Running Action Extractor Standalone

You can also run the action-driven clustering component independently to analyze transcript clusters:

```bash
python -m src.cqg_pipeline.action_extractor --input data/final_transcripts_domain_corrected.json
```

### Action Extractor Options
- `--input`: Path to input JSON file (default: `data/final_transcripts_domain_corrected.json`)
- `--clusters`: Number of clusters to create (default: from `Config.N_CLUSTERS`)
- `--domain`: Filter transcripts by domain (e.g., `--domain Hotel`)

Example with domain filtering:
```bash
python -m src.cqg_pipeline.action_extractor --input data/final_transcripts_domain_corrected.json --domain Hotel --clusters 10
```

This will analyze the transcripts, create clusters, and display discovered action patterns with their characteristics.

## Output
- **Queries**: `output/task1_queries.csv` (List of generated queries)
- **Metrics**: `output/task1_metrics.json` (Cluster stats, execution time)

## Configuration
Adjust settings in `src/cqg_pipeline/config.py`:
- `N_CLUSTERS`: Number of action clusters to find (default: 8).
- `N_QUERIES_PER_SEED`: Queries generated per cluster theme.
- `LLM_MODEL`: Model to use (default: `groq/llama-3.3-70b-versatile`).
