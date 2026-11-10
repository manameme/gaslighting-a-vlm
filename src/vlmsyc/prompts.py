"""Build the two-turn conversation: POPE question, model answer, user pushback.

Messages use the Hugging Face chat format with typed content, which every
candidate model's processor turns into its own chat template:
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from .data import Question

ALLOWED_FIELDS = {"object", "claim", "Claim"}


@dataclass(frozen=True)
class Pushback:
    """One pushback template: a type (e.g. 'assertive') and a paraphrase number."""

    type: str
    paraphrase_id: int
    template: str


def pushbacks_from_config(
    cfg: dict, types: list[str] | None = None, paraphrase_ids: list[int] | None = None
) -> list[Pushback]:
    """All (type, paraphrase) templates from configs/pushback.yaml, optionally filtered."""
    out = []
    for ptype, templates in cfg["pushback"].items():
        if types is not None and ptype not in types:
            continue
        for pid, template in enumerate(templates):
            if paraphrase_ids is not None and pid not in paraphrase_ids:
                continue
            fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
            if not fields <= ALLOWED_FIELDS:
                raise ValueError(f"Template {ptype}[{pid}] uses unknown fields {fields - ALLOWED_FIELDS}")
            out.append(Pushback(ptype, pid, template))
    return out


def article(obj: str) -> str:
    """'a' or 'an'. A vowel-letter rule is enough for the 79 POPE objects."""
    return "an" if obj[0].lower() in "aeiou" else "a"


def claim_text(obj: str, a1: str) -> str:
    """The user's claim, which is always the opposite of the model's turn-1 answer."""
    if a1 == "yes":
        return f"there is no {obj} in the image"
    if a1 == "no":
        return f"there is {article(obj)} {obj} in the image"
    raise ValueError(f"a1 must be 'yes' or 'no', got {a1!r}")


def render_pushback(pushback: Pushback, obj: str, a1: str) -> str:
    claim = claim_text(obj, a1)
    return pushback.template.format(object=obj, claim=claim, Claim=claim[0].upper() + claim[1:])


def apply_condition(cfg: dict, condition: str) -> dict:
    """The prompt config for one condition ('base' or a Phase 6 mitigation).

    cfg['conditions'][condition] holds the keys it overrides, e.g.
    {'system': ...} for the grounding prompt, {'relook': ...} for the re-look
    prompt, {'pushback_suffix': ''} for the instruction-placement check.
    """
    conditions = cfg.get("conditions") or {"base": {}}
    if condition not in conditions:
        raise ValueError(f"Unknown condition {condition!r}; known: {sorted(conditions)}")
    return {**cfg, **(conditions[condition] or {}), "condition": condition}


def pushback_message(pushback: Pushback, obj: str, a1: str, cfg: dict) -> str:
    """The full turn-2 user text: pushback [+ re-look request] [+ suffix].

    The suffix (e.g. "Answer yes or no.") repeats the answer format: without
    it, models often reply "I apologise..." and put little probability on
    yes/no at turn 2. It stays last so the answer format is the final words.
    """
    parts = [render_pushback(pushback, obj, a1), cfg.get("relook") or "", cfg.get("pushback_suffix") or ""]
    return " ".join(p.strip() for p in parts if p and p.strip())


def _system(cfg: dict) -> list[dict]:
    """A system message if the condition has one (grounding prompt), else nothing."""
    text = (cfg.get("system") or "").strip()
    return [{"role": "system", "content": [{"type": "text", "text": text}]}] if text else []


def turn1_messages(q: Question, cfg: dict) -> list[dict]:
    """Turn 1: [system,] the image and the POPE question with the yes/no instruction."""
    return _system(cfg) + [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": cfg["turn1"].format(question=q.text)}],
        }
    ]


def turn2_messages(q: Question, a1: str, pushback: Pushback, cfg: dict) -> list[dict]:
    """Turn 2: turn 1, the model's answer a1 as the assistant message, then the pushback.

    In the re-look condition the image is attached again to the pushback message.
    """
    content = [{"type": "text", "text": pushback_message(pushback, q.object, a1, cfg)}]
    if cfg.get("relook"):
        content = [{"type": "image"}] + content
    return turn1_messages(q, cfg) + [
        {"role": "assistant", "content": [{"type": "text", "text": cfg["answer_text"][a1]}]},
        {"role": "user", "content": content},
    ]


def n_images(messages: list[dict]) -> int:
    return sum(c["type"] == "image" for m in messages for c in m["content"])
