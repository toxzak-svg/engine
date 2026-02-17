from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.reporting import build_consolidated_memo, build_final_manifest, load_comparison_reports, load_window_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated ranking memo and final run manifest.")
    parser.add_argument(
        "--marker",
        default="artifacts/memos/full_program_run_marker.txt",
        help="Marker file path used to filter run/comparison artifacts",
    )
    parser.add_argument(
        "--memo-out",
        default="artifacts/memos/consolidated_final_ranking.md",
        help="Output path for markdown memo",
    )
    parser.add_argument(
        "--manifest-out",
        default="artifacts/memos/final_run_manifest.json",
        help="Output path for JSON manifest",
    )
    args = parser.parse_args()

    window = load_window_artifacts(args.marker)
    reports = load_comparison_reports(window.comparison_files)
    memo_text = build_consolidated_memo(reports, marker_iso_utc=window.marker_iso_utc)

    memo_path = Path(args.memo_out)
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text(memo_text, encoding="utf-8")

    manifest = build_final_manifest(window, memo_path=memo_path)
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"memo_path": str(memo_path), "manifest_path": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
