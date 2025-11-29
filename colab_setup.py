"""
Colab Setup Script for CQG Pipeline.

This script helps set up the environment in Google Colab or other notebook environments.
It handles:
1. Installing dependencies
2. Configuring Python path
3. Verifying data availability
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment(
    install_dependencies: bool = True,
    mount_drive: bool = False,
    project_root_env: str = "CQG_ROOT"
):
    """
    Set up the CQG pipeline environment.
    
    Args:
        install_dependencies: Whether to pip install requirements.txt
        mount_drive: Whether to mount Google Drive (Colab only)
        project_root_env: Environment variable to set for project root
    """
    print("🚀 Setting up CQG Pipeline Environment...")
    
    # 1. Determine paths
    current_dir = Path(__file__).resolve().parent
    # Assuming this script is in src/cqg_pipeline/, the repo root is 2 levels up
    # But if we are in Colab and cloned to /content/repo, it might be different.
    # Let's try to find the root by looking for requirements.txt
    
    repo_root = current_dir
    for _ in range(3):
        if (repo_root / "requirements.txt").exists():
            break
        repo_root = repo_root.parent
    
    print(f"📂 Detected Repo Root: {repo_root}")
    
    # 2. Add to sys.path
    # We need to add the parent of 'src' (or the root itself if src is top-level)
    # to sys.path so 'import src.cqg_pipeline' or 'from cqg_pipeline' works.
    # Based on the structure: /path/to/repo/src/cqg_pipeline
    # We want to be able to do `from cqg_pipeline import ...` if we are inside src?
    # Or `import cqg_pipeline`?
    # The user's error was "No module named 'src'". This suggests they might be doing `from src.cqg_pipeline...`
    # If the structure is repo/src/cqg_pipeline, then adding 'repo' to path allows 'import src...'
    # Adding 'repo/src' to path allows 'import cqg_pipeline...'
    
    # Let's add both to be safe, but prioritize the one that matches the import style.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        print(f"✅ Added {repo_root} to sys.path")
        
    src_path = repo_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
        print(f"✅ Added {src_path} to sys.path")

    # 3. Set Environment Variable for Config
    os.environ[project_root_env] = str(repo_root)
    print(f"✅ Set {project_root_env}={repo_root}")

    # 4. Install Dependencies
    if install_dependencies:
        req_path = repo_root / "requirements.txt"
        if not req_path.exists():
            # Try looking in the same dir as this script
            req_path = current_dir / "requirements.txt"
        
        if req_path.exists():
            print(f"📦 Installing dependencies from {req_path}...")
            
            # Check if uv is available
            use_uv = False
            try:
                subprocess.run(["uv", "--version"], check=True, capture_output=True)
                use_uv = True
                print("🚀 Found 'uv', using it for faster installation!")
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            if use_uv:
                # Use uv pip install
                # In Colab, we might need --system if not in venv, but uv handles this well usually
                # or we can just run `uv pip install -r ... --system`
                cmd = ["uv", "pip", "install", "-r", str(req_path), "--system"]
                subprocess.check_call(cmd)
            else:
                # Fallback to standard pip
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req_path)])
                
            print("✅ Dependencies installed.")
        else:
            print("⚠️  requirements.txt not found. Skipping installation.")

    # 5. Mount Drive (Optional)
    if mount_drive:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive mounted.")
        except ImportError:
            print("⚠️  Not running in Google Colab or drive module missing.")

    print("\n✨ Setup Complete! You can now import modules.")
    print("Example: from cqg_pipeline.pipeline_runner import CQGPipeline")

if __name__ == "__main__":
    setup_environment()
