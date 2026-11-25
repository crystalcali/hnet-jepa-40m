"""Generate from a safetensors checkpoint."""
import argparse
import json
from pathlib import Path
import unicodedata

from safetensors.torch import load_file
from tokenizers import Tokenizer
import torch

from hnet.data import Codec, pad
from hnet.model import build_model
from hnet.training import amp


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    config = json.loads((checkpoint / "config.json").read_text())
    model = build_model(config, device)
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")))
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json")) if config["model"] == "A" else None
    return model.eval(), Codec(tokenizer), config


def sample(logits, temperature, top_p):
    if temperature == 0:
        return int(logits.argmax())
    values, ids = (logits.float() / temperature).sort(descending=True)
    probs = values.softmax(-1)
    keep = probs.cumsum(0) - probs <= top_p
    return int(ids[torch.multinomial(probs * keep, 1)])


def prefix_is_utf8(data):
    import codecs
    try:
        codecs.getincrementaldecoder("utf8")("strict").decode(data, final=False)
        return True
    except UnicodeDecodeError:
        return False


@torch.inference_mode()
def generate(model, codec, prompt, language, *, source_language=None, mode="lm", iterations=1,
             max_bytes=128, temperature=0., top_p=.95, utf8=False):
    if not 0 < max_bytes or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid decoding parameters")
    text = unicodedata.normalize("NFC", prompt)
    device = next(model.parameters()).device
    source_language = source_language or language
    conditional = mode != "lm"
    task = mode == "task"
    context_cost = len(text.encode()) + (6 if task else 5 if conditional else 2)
    max_bytes = min(max_bytes, 8192 - context_cost)
    if max_bytes < 1:
        raise ValueError("No context capacity for answer")
    base = codec.ids(text, source_language if conditional else language, eos=conditional, task=task)
    memory = valid = None
    if conditional and codec.tokenizer is None:
        with amp(device):
            encoded = model.encode(pad([base], codec.pad, device), terminal=True)
            memory = model.reason(encoded.latents[-1], iterations)
            valid = encoded.valid
    target = codec.ids("", language, eos=False)
    prefix = target if conditional and codec.tokenizer is None else base + (target if conditional else [])
    generated = []
    for _ in range(max_bytes):
        ids = pad([prefix + generated], codec.pad, device)
        with amp(device):
            if memory is not None:
                logits = model.condition_decode(model.encode(ids), memory, valid)[0, -1]
            else:
                logits = model(ids, K=iterations)[0, -1]
        logits = logits.float().clone()
        if codec.tokenizer is not None:
            vocab = codec.tokenizer.get_vocab_size()
            logits[vocab:] = -torch.inf
            for marker in ("<pad>", "<bos>", "<en>", "<ru>", "<task>", "<sep>"):
                logits[codec.tokenizer.token_to_id(marker)] = -torch.inf
        elif utf8:
            raw = bytes(generated)
            if raw.decode("utf-8", errors="ignore").encode() != raw:
                logits[codec.eos] = -torch.inf
            for token in range(256):
                if not prefix_is_utf8(raw + bytes([token])):
                    logits[token] = -torch.inf
        token = sample(logits, temperature, top_p)
        if token == codec.eos:
            break
        candidate = generated + [token]
        if codec.tokenizer is not None and len(codec.decode(candidate).encode()) > max_bytes:
            break
        generated = candidate
    if codec.tokenizer is None:
        raw = bytes(generated)
        try:
            return {"text": raw.decode("utf-8"), "valid_utf8": True, "bytes": len(raw)}
        except UnicodeDecodeError:
            return {"text": raw.decode("utf-8", errors="replace"), "valid_utf8": False, "bytes": len(raw)}
    text = codec.decode(generated)
    return {"text": text, "valid_utf8": "\ufffd" not in text, "bytes": len(text.encode())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("lm", "task", "translation", "continuation"), default="lm")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--language", choices=("ru", "en"), default="ru")
    parser.add_argument("--source-language", choices=("ru", "en"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-bytes", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top-p", type=float, default=.95)
    parser.add_argument("--utf8-constraint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, codec, _ = load_model(args.checkpoint, torch.device(args.device))
    print(json.dumps(generate(model, codec, args.prompt, args.language, source_language=args.source_language,
                              mode=args.mode, iterations=args.iterations, max_bytes=args.max_bytes,
                              temperature=args.temperature, top_p=args.top_p, utf8=args.utf8_constraint),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
