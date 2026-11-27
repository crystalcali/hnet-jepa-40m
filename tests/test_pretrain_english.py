import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from data.pretrain_english import Windows, tokenize, window_ranges
from hnet.model import count_parameters
from hnet.pretrain import build_pretrain_model, ratio_loss
from pretrain_en import ByteBPEBatches, WindowSampler, evaluate, get_corpus

CPU = torch.device("cpu")
SMALL = {"width": 64, "blocks": 1, "ff": 96, "outer_width": 32, "outer_depth": 1}


def test_default_experiment_parameter_counts():
    cfg = yaml.safe_load(Path("configs/pretrain_en.yaml").read_text())
    a = build_pretrain_model("bpe", cfg, cfg["bpe_vocab"], CPU)
    b = build_pretrain_model("hnet", cfg, cfg["bpe_vocab"], CPU)
    assert count_parameters(a) == 38_545_920
    assert count_parameters(b) == 38_498_976
    assert count_parameters(a.blocks) == count_parameters(b.blocks)
    assert a.embed.weight is not a.head.weight


def test_hnet_causal_routing_padding_and_gradients():
    torch.manual_seed(42)
    model = build_pretrain_model("hnet", SMALL, 8192, CPU)
    model.router.threshold.fill_(.32)
    ids = torch.tensor([[256, 65, 66, 67, 68, 69, 70], [256, 78, 99, 258, 258, 258, 258]])
    valid = ids.ne(258)
    changed = ids.clone()
    changed[0, 4:] = torch.tensor([97, 99, 120])
    original = model(ids, valid)
    other = model(changed, valid)
    short = model(ids[:, :4], valid[:, :4])
    assert not (original.route.boundaries & ~valid).any()
    assert original.route.boundaries[:, 0].all()
    torch.testing.assert_close(original.logits[:, :4], other.logits[:, :4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(original.logits[:, :4], short.logits, atol=1e-5, rtol=1e-5)

    labels = torch.tensor([[65, 66, 67, 68, 69, 70, 257], [78, 99, 257, -100, -100, -100, -100]])
    ce = F.cross_entropy(original.logits.flatten(0, 1), labels.flatten())
    ce_grad = torch.autograd.grad(ce, model.router.query.weight, retain_graph=True)[0]
    router_grad = torch.autograd.grad(ratio_loss(original.route, 4.), model.router.key.weight)[0]
    assert ce_grad.abs().sum() > 0 and router_grad.abs().sum() > 0


def test_utf8_windows_are_lossless():
    raw = "abcé★🙂text".encode()
    parts = [raw[a:b] for a, b in window_ranges(raw, 4)]
    assert b"".join(parts) == raw
    assert all(0 < len(part) <= 4 and part.decode() for part in parts)


@pytest.fixture
def corpus(tmp_path):
    texts = ["North winds cross the green valley.", "The electric light flickers.",
             "A short example of English web text.", "Music, theatre and cafés.", "A quiet street."]
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps({"text": line}) + "\n" for line in texts), encoding="utf-8"
        )
    tokenize(tmp_path, 280, 16, 32)
    return tmp_path


def test_same_source_windows_and_resumable_order(corpus):
    info, tokenizer, windows = get_corpus(corpus, "train")
    bpe, byte = (ByteBPEBatches(tokenizer, kind, CPU) for kind in ("bpe", "hnet"))
    rows = windows.read([0, 1])
    ai, av, ay, ac = bpe(rows)
    bi, bv, by, bc = byte(rows)
    assert ac == bc
    assert bv.sum() == sum(map(len, rows)) + len(rows)
    assert av.sum() == ac + len(rows)

    sampler = WindowSampler(windows, 2, 2, 42)
    running = iter(sampler)
    next(running)
    saved = sampler.snapshot()
    restored = WindowSampler(windows, 2, 2, 42)
    restored.restore(saved)
    assert list(running) == list(restored)

    for mode, batches in (("bpe", bpe), ("hnet", byte)):
        model = build_pretrain_model(mode, SMALL, info["vocab"], CPU)
        found = evaluate(model, batches, windows, CPU, 0, 2)
        assert found["bytes"] == sum(stop - start for _, start, stop in windows.index)
        assert 0 < found["bpb"]["en"] < 20
