"""Scorers: turn (model, example, generation settings) into a per-example result dict with a ``score`` in [0,1].

Each example costs exactly one forward pass per sample in budget accounting.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import torch

from ..util import WSPError


@dataclass
class GenerationSettings:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 64
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    thinking: bool = False
    seed: int = 0
    samples: int = 1
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> GenerationSettings:
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__ and k != "extra"}
        s = cls(**known)
        s.extra.update(d.pop("extra", {}))
        s.extra.update(d)
        if s.samples < 1 or s.samples > 64:
            raise WSPError("samples must be 1..64", code="bad_arg")
        return s

    def to_dict(self) -> dict:
        return asdict(self)


def detect_loop(ids: list[int], min_repeats: int = 3, min_span: int = 24, max_period: int = 64) -> bool:
    """True if the tail of `ids` is a block of period p repeated >= min_repeats times covering >= min_span tokens."""
    n = len(ids)
    for p in range(1, min(max_period, n // min_repeats) + 1):
        reps = 1
        i = n - p
        while i - p >= 0 and ids[i - p:i] == ids[i:i + p]:
            reps += 1
            i -= p
        if reps >= min_repeats and reps * p >= min_span:
            return True
    return False


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    nums = _NUM.findall(text)
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth = 1
    out = []
    while j < len(text) and depth:
        c = text[j]
        depth += (c == "{") - (c == "}")
        if depth:
            out.append(c)
        j += 1
    return "".join(out).strip()


def _norm_answer(a: str | None) -> str:
    if a is None:
        return ""
    a = a.strip().replace(" ", "").replace("\\!", "").replace("$", "")
    a = re.sub(r"\\text\{(.*?)\}", r"\1", a)
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    try:
        f = float(a)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return a


def _build_prompt(ad, ex: dict, s: GenerationSettings):
    if "chat" in ex or ex.get("as_chat"):
        msgs = ex.get("chat") or [{"role": "user", "content": ex["prompt"]}]
        tok = ad.tokenizer
        if tok is None or not getattr(tok, "chat_template", None):
            raise WSPError("chat-style examples need a tokenizer with a chat template", code="unsupported")
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=s.thinking,
                                          return_tensors="pt")
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return ids
    return ad.encode(ex.get("prompt", ex.get("input")))


def score_example(ad, scorer: str, ex: dict, s: GenerationSettings) -> dict:
    if scorer == "argmax_acc":
        out = ad.forward(ad.encode(ex["input"])).reshape(-1)
        pred = int(out.argmax())
        return {"score": float(pred == int(ex["target"])), "pred": pred, "target": int(ex["target"])}
    if scorer == "next_token_acc":
        enc = ad.encode(ex["input"])
        logits = ad.forward(enc)[0]
        T = enc.shape[1]
        a, b = int(ex.get("score_from", 1)), int(ex.get("score_to", T - 1))
        pred = logits[a:b].argmax(-1)
        tgt = enc[0, a + 1:b + 1]
        c = (pred == tgt).float()
        return {"score": float(c.mean()) if c.numel() else 0.0, "correct": int(c.sum()), "total": int(c.numel())}
    if scorer == "seq_target_acc":
        enc = ad.encode(ex["input"])
        pred = ad.forward(enc)[0].argmax(-1)
        a = int(ex.get("score_from", 1))
        tgt = torch.tensor(ex["targets"])
        c = (pred[a:len(tgt)] == tgt[a:]).float()
        return {"score": float(c.mean()) if c.numel() else 0.0, "correct": int(c.sum()), "total": int(c.numel())}
    if scorer.startswith("gen_"):
        enc = _build_prompt(ad, ex, s)
        gens = [ad.generate(enc, {**s.to_dict(), "seed": s.seed + k}) for k in range(s.samples)]
        return _score_generations(scorer, ex, gens)
    raise WSPError(f"unknown scorer {scorer!r}", code="bad_suite")


def _score_generations(scorer: str, ex: dict, gens: list[dict]) -> dict:
    scores, hits, loops, lens, texts = [], [], [], [], []
    for g in gens:
        text, ids = g["text"], g["new_ids"]
        looped = detect_loop(ids)
        if scorer == "gen_contains":
            sc = float(str(ex["target"]).lower() in text.lower())
        elif scorer == "gen_numeric":
            sc = float(_norm_answer(last_number(text)) == _norm_answer(str(ex["target"])))
        elif scorer == "gen_boxed":
            sc = float(_norm_answer(last_boxed(text)) == _norm_answer(str(ex["target"])))
        elif scorer == "gen_regex":
            sc = float(re.search(ex["pattern"], text) is not None)
        elif scorer == "gen_loop":
            sc = float(not looped and not g["hit_token_limit"])
        else:
            raise WSPError(f"unknown scorer {scorer!r}", code="bad_suite")
        scores.append(sc)
        hits.append(float(g["hit_token_limit"]))
        loops.append(float(looped))
        lens.append(len(ids))
        texts.append(text[-400:])
    n = len(scores)
    return {"score": sum(scores) / n, "hit_token_limit": sum(hits) / n, "looped": sum(loops) / n,
            "new_tokens": sum(lens) / n, "sample_tails": texts}


def score_examples(ad, scorer: str, examples: list[dict], s: GenerationSettings, batch_size: int = 1,
                   progress=None) -> list[dict]:
    """Score many examples. Generation scorers are batched when the adapter supports it (big speed-up on GPU)."""
    if not (scorer.startswith("gen_") and batch_size > 1 and hasattr(ad, "generate_batch")):
        out = []
        for i, ex in enumerate(examples):
            out.append(score_example(ad, scorer, ex, s))
            if progress:
                progress(i + 1, len(examples))
        return out
    results: list[dict] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        encs = [_build_prompt(ad, ex, s) for ex in chunk]
        per_ex: list[list[dict]] = [[] for _ in chunk]
        for k in range(s.samples):
            gens = ad.generate_batch(encs, {**s.to_dict(), "seed": s.seed + k})
            for i, g in enumerate(gens):
                per_ex[i].append(g)
        results += [_score_generations(scorer, ex, gs) for ex, gs in zip(chunk, per_ex)]
        if progress:
            progress(len(results), len(examples))
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n": 0, "score": None}
    sc = torch.tensor([r["score"] for r in results], dtype=torch.float64)
    m = {"n": n, "score": float(sc.mean()), "stderr": float(sc.std() / n ** 0.5) if n > 1 else 0.0}
    for key in ("hit_token_limit", "looped", "new_tokens"):
        if key in results[0]:
            m[{"hit_token_limit": "hit_token_limit_rate", "looped": "loop_rate", "new_tokens": "mean_new_tokens"}[key]] = \
                float(sum(r[key] for r in results) / n)
    if "total" in results[0]:
        m["token_accuracy"] = sum(r["correct"] for r in results) / max(1, sum(r["total"] for r in results))
    return m


def passes_for(suite: dict, n_examples: int, s: GenerationSettings) -> int:
    return n_examples * (s.samples if suite["scorer"].startswith("gen_") else 1)
