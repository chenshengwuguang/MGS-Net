"""Create splits for your own paired data."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mgs_net.preparation import split_main

if __name__ == "__main__":
    split_main()
