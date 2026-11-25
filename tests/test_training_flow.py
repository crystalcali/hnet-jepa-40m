import sys

from safetensors.torch import load_file
import torch

from dashboard.server import tail_events
from generate import generate, load_model
from scripts.debug_corpus import make_fixture
import train


def test_dry_smoke_resume_and_export(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    data = tmp_path / "corpus"
    make_fixture(data)
    base = ["train.py", "--config", "configs/debug.yaml", "--data", str(data), "--device", "cpu",
            "--debug-corpus", "--schedule-bpe", "100000"]
    run = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", base + ["--run", str(run), "--max-steps", "1"])
    train.main()
    assert (run / "latest" / "model.safetensors").is_file()
    monkeypatch.setattr(sys, "argv", base + ["--run", str(run), "--max-steps", "2", "--resume", "latest"])
    train.main()
    train_events = [x for x in tail_events(run / "metrics.jsonl") if x["kind"] == "train"]
    assert [event["step"] for event in train_events] == [1, 2]
    assert train_events[0]["bpe"] < train_events[1]["bpe"]
    assert train_events[1]["estimated_training_flops"] > 0
    control = tmp_path / "uninterrupted"
    monkeypatch.setattr(sys, "argv", base + ["--run", str(control), "--max-steps", "2"])
    train.main()
    w_a = load_file(str(run / "latest" / "model.safetensors"))
    w_b = load_file(str(control / "latest" / "model.safetensors"))
    for name in w_a:
        torch.testing.assert_close(w_a[name], w_b[name], rtol=0, atol=0)
    model, codec, _ = load_model(run / "latest", torch.device("cpu"))
    predicted = generate(model, codec, "Test", "en", mode="task", max_bytes=3)
    assert predicted["bytes"] <= 3
