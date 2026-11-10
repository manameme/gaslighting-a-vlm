"""Two-turn pipeline: ask, read a1, push back, read a2, write JSONL rows.

One output row per (uid, model, pushback_type, paraphrase_id, condition).
Turn 1 is computed once per question and shared by all its pushbacks.

Robustness for Kaggle:
  - rows are appended and flushed (and fsynced) after every batch, so a session
    that dies loses at most one batch;
  - --resume skips rows already in the file;
  - --shard k/n gives each teammate a disjoint slice of the questions, each
    written to its own file, merged later by scripts/merge_shards.py;
  - a GPU out-of-memory error halves the batch size and retries.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable, Iterable

from .data import Question, load_image
from .models import VLM, YesNo
from .prompts import Pushback, apply_condition, n_images, pushback_message, turn1_messages, turn2_messages

KEY_FIELDS = ("uid", "model", "pushback_type", "paraphrase_id", "condition")


# ---------- sharding ----------


def parse_shard(text: str) -> tuple[int, int]:
    """'k/n' -> (k, n), with 0 <= k < n. Shards are numbered from 0."""
    try:
        k, n = (int(x) for x in text.split("/"))
    except ValueError:
        raise ValueError(f"--shard must look like k/n, e.g. 0/3; got {text!r}") from None
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"--shard needs 0 <= k < n; got {text!r}")
    return k, n


def select_shard(questions: list[Question], k: int, n: int) -> list[Question]:
    """Every n-th question starting at k. Shards are disjoint and together cover everything."""
    return [q for i, q in enumerate(questions) if i % n == k]


def output_path(results_dir: Path, kind: str, partition: str, model_name: str, k: int, n: int) -> Path:
    """e.g. RESULTS_DIR/baseline_dev_qwen2vl_2b_shard0of3.jsonl. The merged file drops the shard part."""
    return Path(results_dir) / f"{kind}_{partition}_{model_name}_shard{k}of{n}.jsonl"


# ---------- JSONL helpers ----------


def row_key(row: dict) -> tuple:
    return tuple(row[f] for f in KEY_FIELDS)


def read_rows(path: str | Path) -> list[dict]:
    """Data rows of a JSONL file. Header lines ({"_meta": ...}) and a torn last line are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut off when a session died
            if "_meta" not in obj:
                rows.append(obj)
    return rows


def read_metas(path: str | Path) -> list[dict]:
    """The {"_meta": ...} header lines of a JSONL file (one per run that appended to it)."""
    path = Path(path)
    if not path.exists():
        return []
    metas = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" in obj:
                metas.append(obj["_meta"])
    return metas


# Settings that change the prompts or model but not the row keys. Resuming a
# file with a different value would silently mix two experiments in one file.
META_MUST_MATCH = ("pushback_suffix", "model_config", "condition", "system", "relook")


def check_meta_consistent(path: str | Path, meta: dict) -> None:
    for old in read_metas(path):
        for key in META_MUST_MATCH:
            if key in old and key in meta and old[key] != meta[key]:
                raise ValueError(
                    f"{path} was written with {key}={old[key]!r}, but this run has {meta[key]!r}. "
                    "Use a different output file."
                )


def read_done_keys(path: str | Path) -> set[tuple]:
    return {row_key(r) for r in read_rows(path)}


def append_jsonl(path: str | Path, objs: Iterable[dict]) -> None:
    """Append lines and force them to disk, so a crash loses at most this batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        # A torn last line from a killed session would merge with our first
        # line, so start on a fresh line if the file does not end with one.
        if f.tell() > 0:
            with open(path, "rb") as r:
                r.seek(-1, os.SEEK_END)
                if r.read(1) != b"\n":
                    f.write("\n")
        for obj in objs:
            f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------- running the model ----------


class BatchRunner:
    """Calls model.batch_yes_no in chunks, halving the chunk size on out-of-memory.

    The reduced size is kept for later calls, so we do not hit the same error
    on every batch.
    """

    def __init__(self, model: VLM, batch_size: int):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, messages_list: list[list[dict]], images: list) -> list[YesNo]:
        out: list[YesNo] = []
        i = 0
        while i < len(messages_list):
            size = self.batch_size
            try:
                out.extend(self.model.batch_yes_no(messages_list[i : i + size], images[i : i + size]))
                i += size
            except Exception as err:
                if not self.model.is_oom(err) or size == 1:
                    raise
                self.model.free_memory()
                self.batch_size = max(1, size // 2)
                print(f"[oom] batch size {size} -> {self.batch_size}", flush=True)
        return out


def run_questions(
    model: VLM,
    questions: list[Question],
    pushbacks: list[Pushback],
    prompt_cfg: dict,
    *,
    partition: str,
    condition: str = "base",
    batch_size: int = 8,
    image_loader: Callable[[Question], object] | None = None,
    seed: int = 0,
    git_commit: str = "unknown",
    runner: BatchRunner | None = None,
) -> list[dict]:
    """Rows for every (question, pushback). Turn 1 runs once per question."""
    runner = runner or BatchRunner(model, batch_size)
    image_loader = image_loader or (lambda q: load_image(q.image_path))
    images = {q.uid: image_loader(q) for q in questions}

    # Turn 1: one prompt per question.
    t1 = runner([turn1_messages(q, prompt_cfg) for q in questions], [images[q.uid] for q in questions])
    first = {q.uid: r for q, r in zip(questions, t1)}

    # Turn 2: one prompt per (question, pushback), using that question's a1.
    pairs = [(q, pb) for q in questions for pb in pushbacks]
    t2_messages = [turn2_messages(q, first[q.uid].answer, pb, prompt_cfg) for q, pb in pairs]
    # One copy of the image per image slot (two in the re-look condition).
    t2_images = [
        images[q.uid] if n_images(m) == 1 else [images[q.uid]] * n_images(m)
        for (q, _), m in zip(pairs, t2_messages)
    ]
    t2 = runner(t2_messages, t2_images)

    rows = []
    for (q, pb), r2 in zip(pairs, t2):
        r1 = first[q.uid]
        a1, a2 = r1.answer, r2.answer
        rows.append(
            {
                "uid": q.uid,
                "question_id": q.question_id,
                "image_id": q.image_id,
                "pope_split": q.pope_split,
                "partition": partition,
                "model": model.name,
                "pushback_type": pb.type,
                "paraphrase_id": pb.paraphrase_id,
                "pushback_text": pushback_message(pb, q.object, a1, prompt_cfg),
                "condition": condition,
                "object": q.object,
                "label": q.label,
                "p1_yes": r1.p_yes,
                "a1": a1,
                "a1_correct": a1 == q.label,
                "p2_yes": r2.p_yes,
                "a2": a2,
                "a2_correct": a2 == q.label,
                "flip": a2 != a1,
                "yesno_mass_t1": r1.mass,
                "yesno_mass_t2": r2.mass,
                "logp1_yes": r1.logp_yes,
                "logp1_no": r1.logp_no,
                "logp2_yes": r2.logp_yes,
                "logp2_no": r2.logp_no,
                "seed": seed,
                "git_commit": git_commit,
            }
        )
    return rows


def run_question(model: VLM, q: Question, pushback: Pushback, prompt_cfg: dict, condition: str = "base", **kw) -> dict:
    """SPEC section 5 signature: the row for one question and one pushback."""
    return run_questions(model, [q], [pushback], prompt_cfg, condition=condition, **kw)[0]


def prepare_dev_run(experiment_path: str | Path, model_cfg: dict, stage: str, condition: str = "base") -> dict:
    """Everything a dev run needs, shared by pilot.py, run_baseline.py and the notebook.

    stage is 'pilot' or 'baseline' (a section of configs/experiment.yaml).
    Always uses the dev partition: the test split is only for run_final.py.
    """
    from .paths import data_dir, git_commit, load_yaml, package_versions
    from .prompts import pushbacks_from_config
    from .splits import load_questions

    exp = load_yaml(experiment_path)
    data_cfg = load_yaml(exp["data"])
    prompt_cfg = apply_condition(load_yaml(exp["prompts"]), condition)
    questions = load_questions(
        "dev",
        data_dir() / data_cfg["splits_file"],
        data_dir() / data_cfg["pope_dir"],
        data_dir() / data_cfg["image_dir"],
    )
    max_side = model_cfg.get("max_image_side")
    commit = git_commit()
    return {
        "exp": exp,
        "prompt_cfg": prompt_cfg,
        "questions": questions,
        "pushbacks": pushbacks_from_config(prompt_cfg, paraphrase_ids=exp[stage]["paraphrase_ids"]),
        "image_loader": lambda q: load_image(q.image_path, max_side),
        "seed": exp["seed"],
        "git_commit": commit,
        "meta": {
            "stage": stage, "git_commit": commit, "seed": exp["seed"], "versions": package_versions(),
            "pushback_suffix": prompt_cfg.get("pushback_suffix", ""), "model_config": model_cfg,
            "condition": condition, "system": prompt_cfg.get("system"), "relook": prompt_cfg.get("relook"),
        },
        "condition": condition,
    }


def run_to_file(
    model: VLM,
    questions: list[Question],
    pushbacks: list[Pushback],
    prompt_cfg: dict,
    out_path: str | Path,
    *,
    partition: str,
    resume: bool,
    batch_size: int,
    condition: str = "base",
    max_new_questions: int | None = None,
    image_loader: Callable[[Question], object] | None = None,
    seed: int = 0,
    git_commit: str = "unknown",
    meta: dict | None = None,
    log_every: int = 1,
) -> dict:
    """Run every question not yet complete in out_path, appending after each batch.

    A question is complete when all its (pushback, condition) rows are in the
    file. Returns counts and timing. 'remaining' is the number of questions
    still incomplete afterwards (0 means this file is finished).
    """
    out_path = Path(out_path)
    if out_path.exists() and not resume and read_rows(out_path):
        raise FileExistsError(f"{out_path} already has rows. Use --resume, or move the file away.")
    check_meta_consistent(out_path, meta or {})
    done = read_done_keys(out_path)

    def expected(q):
        return {(q.uid, model.name, pb.type, pb.paraphrase_id, condition) for pb in pushbacks}

    todo = [q for q in questions if not expected(q) <= done]
    total_todo = len(todo)
    if max_new_questions is not None:
        todo = todo[:max_new_questions]

    stats = {"questions_done_before": len(questions) - total_todo, "questions_run": 0, "rows_written": 0, "seconds": 0.0}
    if todo:
        append_jsonl(out_path, [{"_meta": {**(meta or {}), "started": time.strftime("%Y-%m-%d %H:%M:%S")}}])

    runner = BatchRunner(model, batch_size)
    start = time.time()
    for b in range(0, len(todo), batch_size):
        batch = todo[b : b + batch_size]
        batch_start = time.time()
        rows = run_questions(
            model, batch, pushbacks, prompt_cfg, partition=partition, condition=condition,
            image_loader=image_loader, seed=seed, git_commit=git_commit, runner=runner,
        )
        # Timing is stored on every row, so it survives resumes and dead sessions.
        sec_per_q = (time.time() - batch_start) / len(batch)
        for r in rows:
            r["batch_sec_per_question"] = sec_per_q
        new_rows = [r for r in rows if row_key(r) not in done]  # skip rows from a half-written question
        append_jsonl(out_path, new_rows)
        done.update(row_key(r) for r in new_rows)
        stats["questions_run"] += len(batch)
        stats["rows_written"] += len(new_rows)
        if (b // batch_size + 1) % log_every == 0:
            elapsed = time.time() - start
            print(
                f"[{out_path.name}] {stats['questions_run']}/{len(todo)} questions this call, "
                f"{elapsed / stats['questions_run']:.2f} s/question",
                flush=True,
            )
    stats["seconds"] = time.time() - start
    stats["remaining"] = total_todo - stats["questions_run"]
    stats["sec_per_question"] = stats["seconds"] / stats["questions_run"] if stats["questions_run"] else None
    return stats
