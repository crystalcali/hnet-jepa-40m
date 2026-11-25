"""Validate an audited offset-addressable cache and optionally export Parquet."""
import argparse
import json
from pathlib import Path

from hnet.data import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    root = args.manifest.parent
    metadata = json.loads((root / "audit.json").read_text())
    if sha256_file(args.manifest) != metadata["manifest_sha256"]:
        raise SystemExit("manifest checksum mismatch")
    for name, identity in metadata["split_sha256"].items():
        if sha256_file(root / f"{name}.jsonl") != identity:
            raise SystemExit(f"{name} checksum mismatch")
        if args.parquet:
            import pyarrow.json
            import pyarrow.parquet
            source = root / f"{name}.jsonl"
            if source.stat().st_size:
                args.out.mkdir(exist_ok=True, parents=True)
                table = pyarrow.json.read_json(source)
                pyarrow.parquet.write_table(table, args.out / f"{name}.parquet", compression="zstd")
    print(json.dumps({"validated": str(root), "train_bpe": metadata["train_bpe"]}))


if __name__ == "__main__":
    main()
