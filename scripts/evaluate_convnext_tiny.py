"""
Evaluation script for Convnext Tiny model.

Usage:
    python scripts/evaluate_convnext_tiny.py
    python scripts/evaluate_convnext_tiny.py --experiment latest
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluate_model import main as evaluate_main

def main():
    """Main evaluation function."""
    # Set model name for unified evaluation script
    original_argv = sys.argv
    sys.argv = ['evaluate_model.py', '--model', 'convnext_tiny'] + sys.argv[1:]
    
    try:
        # Call unified evaluation script
        evaluate_main()
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()
