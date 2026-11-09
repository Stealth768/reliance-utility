"""Data loader and deterministic partition generator for AI2D (AI2 Diagrams).

Loads AI2D test split from Hugging Face parquet shards, enforces 4-choice multiple-choice
schema, samples N=500 main validation questions (seed 0) and disjoint N=400 replication
questions (seed 1), saving IDs write-once to results/derived/.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

AI2D_DATASET_ID = "lmms-lab-encoder/ai2d"
AI2D_SPLIT = "test"
AI2D_PARQUET_URLS = [
    "https://huggingface.co/api/datasets/lmms-lab-encoder/ai2d/parquet/default/test/0.parquet",
    "https://huggingface.co/api/datasets/lmms-lab-encoder/ai2d/parquet/default/test/1.parquet",
]


def get_dataset_metadata() -> Dict[str, str]:
    """Return dataset identifier and split for AI2D reproducibility."""
    return {
        "dataset_id": AI2D_DATASET_ID,
        "split": AI2D_SPLIT,
        "license": "CC BY-SA 4.0",
    }


def fetch_all_ai2d_records(load_images: bool = True) -> List[Dict[str, Any]]:
    """Fetch all records from AI2D test split.

    Parameters
    ----------
    load_images : bool, default=True
        Whether to decode image bytes into PIL Images. If False, 'image' field is None.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts: {'question_id', 'question', 'choices', 'correct_choice_idx', 'image'}.
    """
    import pandas as pd

    cols = ["question", "options", "answer"]
    if load_images:
        cols.append("image")

    records = []
    global_idx = 0
    for url in AI2D_PARQUET_URLS:
        df = pd.read_parquet(url, columns=cols)
        for _, row in df.iterrows():
            qid = f"ai2d_test_{global_idx:05d}"
            global_idx += 1

            choices = list(row["options"])
            if len(choices) != 4:
                continue

            img = None
            if load_images and "image" in row:
                img_data = row["image"]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                elif isinstance(img_data, Image.Image):
                    img = img_data.convert("RGB")

            records.append({
                "question_id": qid,
                "question": str(row["question"]),
                "choices": choices,
                "correct_choice_idx": int(row["answer"]),
                "image": img,
            })
    return records


def sample_ai2d_main_ids(
    dest_path: str = "results/derived/ai2d_main_ids.json",
    n_sample: int = 500,
    seed: int = 0,
) -> List[str]:
    """Sample N=500 question IDs deterministically and write once to dest_path."""
    target_path = REPO_ROOT / dest_path if not Path(dest_path).is_absolute() else Path(dest_path)
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data["question_ids"])

    records = fetch_all_ai2d_records(load_images=False)
    total_len = len(records)
    if n_sample > total_len:
        raise ValueError(f"Requested {n_sample} items, but only {total_len} available in AI2D.")

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(total_len, size=n_sample, replace=False)
    chosen_idx.sort()
    main_qids = [records[i]["question_id"] for i in chosen_idx]

    from src.io_utils import write_raw
    record = {
        "dataset_id": AI2D_DATASET_ID,
        "split": AI2D_SPLIT,
        "n_sample": len(main_qids),
        "seed": seed,
        "question_ids": main_qids,
    }
    write_raw(target_path, record)
    print(f"Sampled N={len(main_qids)} AI2D main items and wrote to {target_path}")
    return main_qids


def sample_ai2d_rep_ids(
    main_ids_path: str = "results/derived/ai2d_main_ids.json",
    dest_path: str = "results/derived/ai2d_rep_ids.json",
    n_sample: int = 400,
    seed: int = 1,
) -> List[str]:
    """Sample N=400 question IDs disjoint from main_ids_path and write once to dest_path."""
    target_path = REPO_ROOT / dest_path if not Path(dest_path).is_absolute() else Path(dest_path)
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data["question_ids"])

    main_qids = set(sample_ai2d_main_ids(main_ids_path))
    records = fetch_all_ai2d_records(load_images=False)
    available_qids = [r["question_id"] for r in records if r["question_id"] not in main_qids]

    if n_sample > len(available_qids):
        raise ValueError(
            f"Requested {n_sample} disjoint items, but only {len(available_qids)} remain in AI2D."
        )

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(len(available_qids), size=n_sample, replace=False)
    chosen_idx.sort()
    rep_qids = [available_qids[i] for i in chosen_idx]

    from src.io_utils import write_raw
    record = {
        "dataset_id": AI2D_DATASET_ID,
        "split": AI2D_SPLIT,
        "n_sample": len(rep_qids),
        "seed": seed,
        "main_ids_path": str(main_ids_path),
        "question_ids": rep_qids,
    }
    write_raw(target_path, record)
    print(f"Sampled N={len(rep_qids)} AI2D disjoint replication items and wrote to {target_path}")
    return rep_qids


def load_ai2d_dataset(
    split: str = "main",
    ids_path: Optional[str] = None,
    load_images: bool = True,
) -> List[Dict[str, Any]]:
    """Load sampled AI2D questions with decoded images.

    Parameters
    ----------
    split : str, default="main"
        Either "main" (N=500) or "replication" (N=400).
    ids_path : Optional[str]
        Explicit path to IDs JSON file.
    load_images : bool, default=True
        Whether to load PIL images.

    Returns
    -------
    List[Dict[str, Any]]
        Sampled item dictionaries.
    """
    if ids_path is None:
        if split == "main":
            ids_path = "results/derived/ai2d_main_ids.json"
            target_qids = set(sample_ai2d_main_ids(ids_path))
        elif split in ("rep", "replication"):
            ids_path = "results/derived/ai2d_rep_ids.json"
            target_qids = set(sample_ai2d_rep_ids(dest_path=ids_path))
        else:
            raise ValueError(f"Unknown split: {split}")
    else:
        with open(ids_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        target_qids = set(data["question_ids"])

    records = fetch_all_ai2d_records(load_images=load_images)
    return [r for r in records if r["question_id"] in target_qids]
