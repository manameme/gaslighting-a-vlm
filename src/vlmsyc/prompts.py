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


def turn1_messages(q: Question, cfg: dict) -> list[dict]:
    """Turn 1: the image and the POPE question with the yes/no instruction."""
    return [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": cfg["turn1"].format(question=q.text)}],
        }
    ]


def turn2_messages(q: Question, a1: str, pushback: Pushback, cfg: dict) -> list[dict]:
    """Turn 2: turn 1, the model's answer a1 as the assistant message, then the pushback."""
    return turn1_messages(q, cfg) + [
        {"role": "assistant", "content": [{"type": "text", "text": cfg["answer_text"][a1]}]},
        {"role": "user", "content": [{"type": "text", "text": render_pushback(pushback, q.object, a1)}]},
    ]
