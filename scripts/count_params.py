import argparse
from pathlib import Path

import yaml

from hnet.model import build_model, count_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    model = build_model(config, "cpu")
    total = count_parameters(model)
    auxiliaries = ("cross_predictors.", "abl_predictors.", "dyn_predictor.")
    train_only = sum(p.numel() for n, p in model.named_parameters() if n.startswith(auxiliaries))
    for name, child in model.named_children():
        print(f"{name:24} {count_parameters(child):12,}")
    print(f"Total: {total:,}; inference: {total-train_only:,}")
    if total > 40_000_000:
        raise SystemExit("Parameter budget exceeded")


if __name__ == "__main__":
    main()
