# Causal Query Generation (CQG) Pipeline

Generates diverse causal queries from contact center transcripts using **Action-Driven Semantic Clustering**.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
# Optional: Install uv for faster setup
pip install uv && uv pip install -r requirements.txt --system
```

### 2. Configuration
Set your API key (Groq recommended):
```bash
export GROQ_API_KEY=your_key_here
```

## Usage

### Command Line
Run the pipeline directly:
```bash
python -m src.cqg_pipeline.pipeline_runner --domain Hotel
```
**Options:**
- `--input`: Input file path (JSON/CSV)
- `--output-csv`: Output CSV path
- `--domain`: Filter by domain (e.g., "Hotel")

### Python API
```python
from src.cqg_pipeline.pipeline_runner import CQGPipeline

pipeline = CQGPipeline()
queries = pipeline.run(
    transcript_file="data/transcripts.json",
    domain="Hotel"
)
```

### Google Colab
1. **Clone** or **Upload** files to Colab.
2. **Install Deps**: `!pip install -r requirements.txt`
3. **Run**:
   ```python
   # If running from flat directory (/content/cqg_pipeline)
   import sys; sys.path.append('/content/cqg_pipeline')
   from pipeline_runner import CQGPipeline
   
   pipeline = CQGPipeline()
   pipeline.run(transcript_file="transcripts.json")
   ```

## Output
- **Queries**: Saved to CSV (default: `output/task1_queries.csv`)
- **Metrics**: Saved to JSON (default: `output/task1_metrics.json`)
