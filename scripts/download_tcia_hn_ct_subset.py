from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


API_BASE = "https://api.github.com/repos/google-deepmind/tcia-ct-scan-dataset/contents"


def get_json(url: str):
    req = Request(url, headers={"User-Agent": "PrAI-downloader"})
    with urlopen(req) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "PrAI-downloader"})
    with urlopen(req) as r:  # noqa: S310
        out_path.write_bytes(r.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected CT_IMAGE.nrrd cases without cloning whole repo")
    parser.add_argument("--split", choices=["validation", "test", "auto"], default="auto")
    parser.add_argument("--labeler", choices=["radiographer", "oncologist", "auto"], default="radiographer")
    parser.add_argument("--cases", type=str, required=True, help="Comma-separated case IDs, e.g. TCGA-CV-5977,0522c0014")
    parser.add_argument("--out-root", type=Path, default=Path("data/raw/tcia_hn_subset/nrrds"))
    args = parser.parse_args()

    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    if not case_ids:
        raise ValueError("No case IDs provided")

    split_candidates = ["validation", "test"] if args.split == "auto" else [args.split]
    labeler_candidates = ["radiographer", "oncologist"] if args.labeler == "auto" else [args.labeler]

    for case_id in case_ids:
        found = False
        for split in split_candidates:
            for labeler in labeler_candidates:
                case_api = f"{API_BASE}/nrrds/{split}/{labeler}/{case_id}?ref=master"
                try:
                    items = get_json(case_api)
                except Exception:
                    continue

                ct_item = None
                for it in items:
                    if it.get("type") == "file" and it.get("name", "").upper() == "CT_IMAGE.NRRD":
                        ct_item = it
                        break

                if ct_item is None:
                    continue

                download_url = ct_item.get("download_url")
                if not download_url:
                    continue

                out_path = args.out_root / split / labeler / case_id / "CT_IMAGE.nrrd"
                download_file(download_url, out_path)
                print(f"[OK] {case_id} ({split}/{labeler}) -> {out_path}")
                found = True
                break
            if found:
                break

        if not found:
            print(f"[WARN] CT_IMAGE.nrrd not found for case {case_id} in tested split/labeler options")


if __name__ == "__main__":
    main()
