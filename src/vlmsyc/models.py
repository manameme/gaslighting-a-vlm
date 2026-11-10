"""VLM wrappers that return P(yes) vs P(no) for the next token.

We never generate text. Each answer is the argmax of p(yes) vs p(no) at the
next-token position (SPEC section 0), read from the logits:
  - collect token ids for "Yes", " Yes", "yes", " yes" (and the same for No),
    keeping only spellings that are a single token;
  - logp_yes = logsumexp of log-softmax over the yes ids (same for no);
  - p_yes = renormalised over {yes, no};
  - mass = total probability on all those tokens. A low mass means the model
    wanted to say something else (e.g. "I apologise"), so its yes/no reading
    is less meaningful. We log it and check it in the pilot.

HFVLM imports torch and transformers only when built, so the CPU tests (which
use MockVLM) need neither package.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass

import numpy as np

YES_VARIANTS = ("Yes", " Yes", "yes", " yes")
NO_VARIANTS = ("No", " No", "no", " no")


@dataclass(frozen=True)
class YesNo:
    """Yes/no reading of one next-token distribution."""

    logp_yes: float  # log P(any yes token), unnormalised over the full vocabulary
    logp_no: float
    p_yes: float  # P(yes) / (P(yes) + P(no))
    mass: float  # P(yes) + P(no): how much the model wanted to answer yes/no at all

    @property
    def answer(self) -> str:
        """Argmax of p(yes) vs p(no). An exact tie goes to 'no'."""
        return "yes" if self.p_yes > 0.5 else "no"


def _logsumexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def yes_no_from_logits(logits: np.ndarray, yes_ids: list[int], no_ids: list[int]) -> YesNo:
    """Turn one row of next-token logits (vocab size,) into a YesNo."""
    x = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        # fp16 overflow shows up here. Fail loudly rather than record garbage.
        raise FloatingPointError("Non-finite logits: try quant: 4bit or smaller images for this model")
    log_probs = x - _logsumexp(x)
    logp_yes = _logsumexp(log_probs[yes_ids])
    logp_no = _logsumexp(log_probs[no_ids])
    p_yes = 1.0 / (1.0 + np.exp(logp_no - logp_yes))
    mass = float(np.exp(np.logaddexp(logp_yes, logp_no)))
    return YesNo(logp_yes=logp_yes, logp_no=logp_no, p_yes=float(p_yes), mass=mass)


def find_yes_no_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Token ids for the yes and no spellings that are a single token.

    Multi-token spellings are skipped: their first token (e.g. a bare space)
    would not mean yes or no on its own.
    """

    def single_ids(variants):
        ids = []
        for v in variants:
            enc = tokenizer.encode(v, add_special_tokens=False)
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        return ids

    yes_ids, no_ids = single_ids(YES_VARIANTS), single_ids(NO_VARIANTS)
    if not yes_ids or not no_ids:
        raise ValueError(f"No single-token yes/no spelling found (yes={yes_ids}, no={no_ids})")
    if set(yes_ids) & set(no_ids):
        raise ValueError("A token id is in both the yes and no sets")
    return yes_ids, no_ids


def images_for_processor(images: list, nested: bool) -> list:
    """Images in the layout the processor expects.

    images[i] is the image of conversation i, or a list of images when the
    conversation shows more than one (the re-look condition shows the image
    again in turn 2). Most processors take one flat list in order of
    appearance; some (SmolVLM) take one list per conversation.
    """
    per_conv = [im if isinstance(im, list) else [im] for im in images]
    return per_conv if nested else [im for conv in per_conv for im in conv]


class VLM:
    """Interface shared by the real model and the mock (SPEC section 3)."""

    name: str = "vlm"

    def render_prompt(self, messages: list[dict]) -> str:
        """The exact text the model sees, after its chat template."""
        raise NotImplementedError

    def batch_yes_no(self, messages_list: list[list[dict]], images: list) -> list[YesNo]:
        raise NotImplementedError

    def yes_no(self, messages: list[dict], image) -> YesNo:
        return self.batch_yes_no([messages], [image])[0]

    def yes_no_logprobs(self, messages: list[dict], image) -> tuple[float, float]:
        """SPEC name: (logp_yes, logp_no) for one prompt."""
        r = self.yes_no(messages, image)
        return r.logp_yes, r.logp_no

    def is_oom(self, error: Exception) -> bool:
        """True if error means 'batch too big for GPU memory'."""
        return False

    def free_memory(self) -> None:
        pass

    def describe(self) -> dict:
        """Facts about the model for result file headers."""
        return {"name": self.name}


class MockOOM(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError in tests."""


class MockVLM(VLM):
    """Deterministic fake model for CPU tests.

    The logits for a prompt come from a random generator seeded by a hash of
    (seed, prompt text), so the same prompt always gives the same answer and
    different prompts give different answers. The image is ignored.
    oom_above simulates GPU memory errors for batches larger than that size.
    """

    def __init__(self, name: str = "mock", seed: int = 0, oom_above: int | None = None):
        self.name = name
        self.seed = seed
        self.oom_above = oom_above
        self.yes_ids, self.no_ids = [0, 1], [2, 3]
        self.vocab_size = 10
        self.batch_sizes_seen: list[int] = []

    def render_prompt(self, messages: list[dict]) -> str:
        lines = []
        for m in messages:
            parts = ["<image>" if c["type"] == "image" else c["text"] for c in m["content"]]
            lines.append(f"<|{m['role']}|> " + " ".join(parts))
        return "\n".join(lines) + "\n<|assistant|>"

    def _logits(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(f"{self.seed}|{text}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return rng.normal(0.0, 2.0, size=self.vocab_size)

    def batch_yes_no(self, messages_list, images):
        self.batch_sizes_seen.append(len(messages_list))
        if self.oom_above is not None and len(messages_list) > self.oom_above:
            raise MockOOM(f"batch {len(messages_list)} > {self.oom_above}")
        return [
            yes_no_from_logits(self._logits(self.render_prompt(m)), self.yes_ids, self.no_ids)
            for m in messages_list
        ]

    def is_oom(self, error):
        return isinstance(error, MockOOM)

    def describe(self):
        return {"name": self.name, "mock": True, "seed": self.seed}


class HFVLM(VLM):
    """A Hugging Face image-text-to-text model, fp16 or 4-bit, on one GPU.

    cfg is a dict from configs/models/<name>.yaml with keys: name, hf_id,
    revision, quant ('none' or '4bit'), processor_kwargs, nested_images.
    """

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("HFVLM needs a CUDA GPU. Use MockVLM for CPU tests.")
        self.torch = torch
        self.cfg = cfg
        self.name = cfg["name"]
        self.device = torch.device("cuda:0")

        self.processor = AutoProcessor.from_pretrained(
            cfg["hf_id"], revision=cfg.get("revision"), **(cfg.get("processor_kwargs") or {})
        )
        # Left padding keeps the last position of every row in a batch the real
        # last token, which is where we read the next-token logits.
        self.processor.tokenizer.padding_side = "left"

        # torch_dtype is accepted by every 4.x release (newer ones also take dtype).
        kwargs = {"torch_dtype": torch.float16, "device_map": {"": 0}}
        if cfg.get("quant") == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )
        elif cfg.get("quant") not in (None, "none"):
            raise ValueError(f"Unknown quant {cfg.get('quant')!r}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg["hf_id"], revision=cfg.get("revision"), **kwargs
        ).eval()

        # Only compute logits for the last position when the model allows it,
        # which saves a lot of memory with a 150k-token vocabulary.
        self.forward_extra = {}
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            self.forward_extra["logits_to_keep"] = 1

        self.yes_ids, self.no_ids = find_yes_no_token_ids(self.processor.tokenizer)

    def render_prompt(self, messages):
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def batch_yes_no(self, messages_list, images):
        torch = self.torch
        texts = [self.render_prompt(m) for m in messages_list]
        imgs = images_for_processor(images, nested=bool(self.cfg.get("nested_images")))
        inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True)
        # Move to GPU. Pixel values become fp16; ids and masks keep their int type.
        inputs = {
            k: (v.to(self.device, dtype=torch.float16) if torch.is_floating_point(v) else v.to(self.device))
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out = self.model(**inputs, **self.forward_extra)
        last = out.logits[:, -1, :].float().cpu().numpy()
        return [yes_no_from_logits(row, self.yes_ids, self.no_ids) for row in last]

    def is_oom(self, error):
        return isinstance(error, self.torch.cuda.OutOfMemoryError)

    def free_memory(self):
        self.torch.cuda.empty_cache()

    def describe(self):
        tok = self.processor.tokenizer
        return {
            "name": self.name,
            "hf_id": self.cfg["hf_id"],
            "revision": self.cfg.get("revision"),
            "quant": self.cfg.get("quant"),
            "gpu": self.torch.cuda.get_device_name(0),
            "yes_tokens": [tok.decode([i]) for i in self.yes_ids],
            "no_tokens": [tok.decode([i]) for i in self.no_ids],
            "config": json.loads(json.dumps(self.cfg)),
        }


def load_model(cfg: dict) -> VLM:
    """Build the model a config describes. name 'mock' gives a MockVLM."""
    if cfg.get("hf_id") == "mock":
        return MockVLM(name=cfg["name"], seed=cfg.get("seed", 0))
    return HFVLM(cfg)
