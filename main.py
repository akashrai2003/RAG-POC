"""
Main entry point for the Asset RAG POC application.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from ui.streamlit_app_simple import main

if __name__ == "__main__":
    main()