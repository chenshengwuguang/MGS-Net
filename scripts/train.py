"""Train MGS-Net."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mgs_net.training import main

if __name__ == "__main__":
    main()
