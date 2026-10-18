"""
src/run_dashboard.py

Launches the local HTTP web server to serve the interactive CRISPR gRNA optimization dashboard.
Delegates to src/serve_dashboard.py.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.serve_dashboard import main as serve_main

def main():
    serve_main()

if __name__ == "__main__":
    main()
