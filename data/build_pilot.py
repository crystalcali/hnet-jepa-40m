"""Build a cache from the pinned RU–EN source adapters."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import yaml

from data.audit import audit
from data.pilot_sources import holdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("cache/pilot_raw"))
    parser.add_argument("--out", type=Path, default=Path("cache/pilot"))
    parser.add_argument("--target-bpe", type=int, default=30_000_000)
    args = parser.parse_args()
    if args.target_bpe < 1:
        parser.error("--target-bpe must be positive")
    config = yaml.safe_load(args.config.read_text())
    for source in config["sources"] + config["heldout_sources"]:
        source["raw_cache"] = str(args.raw.resolve())
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        holdout(config, root / "holdout.jsonl")
        config["benchmark_holdout"] = str(root / "holdout.jsonl")
        resolved = root / "sources.yaml"
        resolved.write_text(yaml.safe_dump(config, allow_unicode=True))
        audit(SimpleNamespace(config=resolved, out=args.out, target_bpe=args.target_bpe,
                              max_rows=None, tokenizer=None))


if __name__ == "__main__":
    main()
