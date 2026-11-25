from pathlib import Path

import pytest
import torch
import yaml

from dashboard.server import tail_events
from hnet.data import Codec, conditional_batch, lm_batch
from hnet.losses import dynamics_index, routing_loss, sigreg, transport
from hnet.model import HNet, PAD, Router, build_model, count_parameters, dechunk
from hnet.training import nll

torch.set_num_threads(1)


def small():
    torch.manual_seed(42)
    model = HNet({"model": "C", "widths": [32, 64, 96, 128], "depths": [1, 1, 1],
                  "blocks": 1, "ff": 96, "checkpoint_core": False}).eval()
    for router in model.routers:
        router.threshold.fill_(.35)
    return model


@pytest.mark.parametrize("name,expected", [
    ("transformer_40m", 39_528_960), ("hnet_40m", 39_922_416), ("hnet_jepa_40m", 39_840_497),
])
def test_parameter_counts(name, expected):
    config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
    assert count_parameters(build_model(config, "cpu")) == expected


def test_prefix_causality_and_padding():
    model = small()
    ids = torch.tensor([[257, 261, 97, 98, 99, 100]])
    altered = ids.clone()
    altered[:, 4:] = torch.tensor([161, 89])
    padded = torch.cat((ids, ids.new_full((1, 2), PAD)), 1)
    with torch.no_grad():
        logits = model(ids, K=2)
        torch.testing.assert_close(logits[:, :4], model(altered, K=2)[:, :4], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(logits, model(padded, K=2)[:, :6], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(logits[:, :4], model(ids[:, :4], K=2), atol=1e-5, rtol=1e-5)


def test_closed_prompt_retains_tail_and_target_is_causal():
    model = small()
    for router in model.routers:
        router.threshold.fill_(1.01)
    source = torch.tensor([[257, 261, 97, 98, 258]])
    changed = source.clone()
    changed[:, -2] = 123
    first = model.encode(source, terminal=True)
    second = model.encode(changed, terminal=True)
    assert all(int(route.boundaries.sum()) == 2 for route in first.routes)
    assert all(int(route.forced.sum()) == 2 for route in first.routes)
    memory = model.reason(first.latents[-1], K=2)
    assert not torch.allclose(memory[:, -1], model.reason(second.latents[-1], K=2)[:, -1])
    prefixes = torch.tensor([[257, 262, 158, 99, 127]])
    logits = model.condition_decode(model.encode(prefixes), memory, first.valid)
    prefixes[:, -1] += 1
    changed_logits = model.condition_decode(model.encode(prefixes), memory, first.valid)
    torch.testing.assert_close(logits[:, :-1], changed_logits[:, :-1])


def test_dechunk_reference_and_ratio_gradients():
    torch.manual_seed(44)
    router = Router(16, threshold=.45)
    x = torch.randn(2, 15, 16, requires_grad=True)
    valid = torch.ones(2, 15, dtype=torch.bool)
    valid[1, 10:] = False
    route = router(x, valid, torch.zeros_like(valid))
    values = torch.randn(*route.index.shape, 9)
    answers = [values[:, 0]]
    for idx in range(1, values.shape[1]):
        p = route.anchors[:, idx, None]
        answers.append(p * values[:, idx] + (1 - p) * answers[-1])
    expected = torch.stack(answers, 1).gather(1, route.parents[..., None].expand(-1, -1, 9))
    torch.testing.assert_close(dechunk(values, route), expected, atol=1e-6, rtol=1e-6)

    model = small()
    cfg = Codec()
    ids, labels, _ = lm_batch(cfg, [{"en": {"text": "A test example."}, "ru": {"text": "Проверка."}}], "cpu")
    encoded = model.encode(ids)
    params = [r.query.weight for r in model.routers]
    ce = nll(model.decode(model.reason(encoded.latents[-1]), encoded), labels)
    r, a, _ = routing_loss(encoded)
    g_ce = torch.autograd.grad(ce, params, retain_graph=True)
    g_ratio = torch.autograd.grad(r + a, params)
    assert sum(float(v.abs().sum()) for v in g_ce) > 0
    assert sum(float(v.abs().sum()) for v in g_ratio) > 0


def test_target_shape_and_sigreg():
    assert [dynamics_index(k, 4) for k in range(1, 9)] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert dynamics_index(1, 1) == 0
    assert dynamics_index(8, 15) == 14
    z = torch.randn(512, 32, requires_grad=True)
    value = sigreg(z, 12)
    torch.testing.assert_close(value, sigreg(z, 12))
    assert value < sigreg(torch.zeros_like(z), 12)
    value.backward()
    assert z.grad.abs().sum() > 0
    plan = transport(torch.zeros(9, 11))
    torch.testing.assert_close(plan.sum(1), torch.full((9,), 1 / 9))
    torch.testing.assert_close(plan.sum(0), torch.full((11,), 1 / 11))


def test_teacher_forced_labels_and_metrics_tail(tmp_path):
    codec = Codec()
    prompt, prefix, labels = conditional_batch(codec, [("A", "Б", "en", "ru", True)], "cpu")
    assert prompt[0, -1] == codec.eos
    assert prefix.tolist() == [[codec.bos, codec.lang("ru"), *"Б".encode()]]
    assert labels.tolist() == [[-100, *"Б".encode(), codec.eos]]
    log = tmp_path / "metrics.jsonl"
    log.write_bytes(b'{"kind":"train","step":1}\n{"kind":"train"')
    assert tail_events(log) == [{"kind": "train", "step": 1}]


def test_rng_fixture_for_recurrence():
    model = small()
    ids = torch.tensor([[257, 261, 109, 110, 120]])
    before = count_parameters(model)
    with torch.no_grad():
        enc = model.encode(ids)
        once = model.reason(enc.latents[-1], K=1)
        twice = model.reason(enc.latents[-1], K=2, trajectory=True)
        torch.testing.assert_close(once, twice[0])
    assert count_parameters(model) == before


def test_objectives_propagate_gradients():
    import random
    from hnet.losses import loss_weights
    from hnet.training import batch_objectives

    class Draw(random.Random):
        def random(self):
            return .1

    model = small()
    for router in model.routers:
        router.threshold.fill_(-.01)
    en, ru = "One plus one is 2.", "Один плюс один = 2."
    row = {"en": {"text": en, "question": "1+1?", "steps": ["Add."], "answer": "2"},
           "ru": {"text": ru, "question": "1+1?", "steps": ["Сложи."], "answer": "2"},
           "alignment": [{"en_span": [0, len(en.encode())], "ru_span": [0, len(ru.encode())],
                          "verified": True}]}
    total, losses, _, _, _ = batch_objectives(
        model, Codec(), [row], 2, loss_weights(.7), Draw(5), torch.device("cpu"), .7, 1.
    )
    assert {"global", "local", "ablation", "dynamics", "translation", "task", "aux"} <= set(losses)
    gradients = torch.autograd.grad(losses["global"] + losses["local"],
                                    tuple(router.query.weight for router in model.routers),
                                    retain_graph=True)
    assert sum(float(grad.abs().sum()) for grad in gradients) > 0.
    total.backward()
    for head in (model.dyn_predictor, model.cross_predictors[0][0], model.abl_predictors[-1]):
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
