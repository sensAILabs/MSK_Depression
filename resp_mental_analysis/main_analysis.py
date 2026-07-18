from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from .step1_full_profile import main as step1_main
from .step2_demo import main as step2_main


def main():

    print("=" * 60)
    print("Resp / Mental analysis started")
    print("=" * 60)

    print("\nStep 1: cohort profile + descriptive figures\n")
    step1_main()

    print("Running step 2 demo...")
    step2_main()

    print("Resp / Mental analysis finished")


if __name__ == "__main__":
    main()