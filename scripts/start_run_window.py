from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a run-window marker for clean manifest/memo generation.")
    parser.add_argument(
        "--out",
        default="artifacts/memos/full_program_run_marker.txt",
        help="Marker output path",
    )
    args = parser.parse_args()

    marker_path = Path(args.out)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    print(str(marker_path.resolve()))


if __name__ == "__main__":
    main()
